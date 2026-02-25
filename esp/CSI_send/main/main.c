/**
 * CSI Send - ESP32C6
 * Sends ESP-NOW packets and receives CSI data acknowledgments
 */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>

#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdmmc_host.h"

// WiFi Configuration - 802.11ax (WiFi 6) with 256 subcarriers
#define WIFI_CHANNEL                11
#define WIFI_BAND_MODE             WIFI_BAND_MODE_2G_ONLY
#define WIFI_BANDWIDTH             WIFI_BW_HT20      // 20MHz = 256 subcarriers
#define WIFI_PROTOCOL              WIFI_PROTOCOL_11AX // WiFi 6 (802.11ax)
#define WIFI_SECOND_CHAN_MODE      WIFI_SECOND_CHAN_NONE  // No secondary channel for 20MHz

// ESP-NOW Configuration
#define ESP_NOW_PHY_MODE           WIFI_PHY_MODE_HE20  // HE20 for 802.11ax 20MHz
#define ESP_NOW_RATE               WIFI_PHY_RATE_MCS0_LGI
#define SEND_FREQUENCY             1  // Hz

// SD Card Configuration
#define SD_MOUNT_POINT             "/sdcard"
#define SD_MISO_PIN                2
#define SD_MOSI_PIN                7
#define SD_CLK_PIN                 6
#define SD_CS_PIN                  18

static const uint8_t SENDER_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_send";

// Must match the fragment structure sent by CSI_recv
typedef struct {
    uint32_t msg_id;
    uint8_t fragment_num;
    uint8_t total_fragments;
    uint16_t fragment_len;
    char data[1460];
} csi_fragment_msg_t;

// Reassembly buffer for one message at a time
#define MAX_REASSEMBLY_BUF  (1460 * 4)  // support up to 4 fragments (~5840 bytes)

typedef struct {
    uint32_t msg_id;
    uint8_t  total_fragments;
    uint8_t  received_mask;   // bitmask of received fragment numbers (up to 8 fragments)
    uint16_t frag_len[8];     // length of each received fragment
    char     buf[MAX_REASSEMBLY_BUF];
} reassembly_ctx_t;

static reassembly_ctx_t s_reassembly = {0};


static esp_err_t sd_card_init(void)
{
    ESP_LOGI(TAG, "Initializing SD card");
    
    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
        .allocation_unit_size = 16 * 1024
    };
    
    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    
    spi_bus_config_t bus_cfg = {
        .mosi_io_num = SD_MOSI_PIN,
        .miso_io_num = SD_MISO_PIN,
        .sclk_io_num = SD_CLK_PIN,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = 4000,
    };
    
    esp_err_t ret = spi_bus_initialize(host.slot, &bus_cfg, SDSPI_DEFAULT_DMA);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Failed to initialize bus.");
        return ret;
    }
    
    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = SD_CS_PIN;
    slot_config.host_id = host.slot;
    
    sdmmc_card_t *card;
    ret = esp_vfs_fat_sdspi_mount(SD_MOUNT_POINT, &host, &slot_config, &mount_config, &card);
    
    if (ret != ESP_OK) {
        if (ret == ESP_FAIL) {
            ESP_LOGE(TAG, "Failed to mount filesystem.");
        } else {
            ESP_LOGE(TAG, "Failed to initialize the card (%s).", esp_err_to_name(ret));
        }
        return ret;
    }
    
    ESP_LOGI(TAG, "SD card mounted successfully");
    sdmmc_card_print_info(stdout, card);
    
    return ESP_OK;
}

static void wifi_init(void)
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
    ESP_ERROR_CHECK(esp_wifi_start());
    ESP_ERROR_CHECK(esp_wifi_set_band_mode(WIFI_BAND_MODE));
    
    wifi_protocols_t protocols = { .ghz_2g = WIFI_PROTOCOL };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(WIFI_IF_STA, &protocols));
    
    wifi_bandwidths_t bandwidth = { .ghz_2g = WIFI_BANDWIDTH };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(WIFI_IF_STA, &bandwidth));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_STA, WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
#endif

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    ESP_ERROR_CHECK(esp_wifi_set_channel(WIFI_CHANNEL, WIFI_SECOND_CHAN_MODE));
    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, SENDER_MAC));
}

static void esp_now_recv_cb(const esp_now_recv_info_t *recv_info, const uint8_t *data, int len)
{
    if (!recv_info || !data) {
        ESP_LOGW(TAG, "Invalid fragment received (null pointer)");
        return;
    }

    // Minimum header: msg_id(4) + fragment_num(1) + total_fragments(1) + fragment_len(2) = 8 bytes
    const int HEADER_SIZE = sizeof(uint32_t) + sizeof(uint8_t) * 2 + sizeof(uint16_t);
    if (len < HEADER_SIZE) {
        ESP_LOGW(TAG, "Packet too short (%d bytes), ignoring", len);
        return;
    }

    const csi_fragment_msg_t *frag = (const csi_fragment_msg_t *)data;

    if (frag->fragment_num >= 8 || frag->total_fragments == 0 || frag->total_fragments > 8) {
        ESP_LOGW(TAG, "Invalid fragment meta: num=%d total=%d",
                 frag->fragment_num, frag->total_fragments);
        return;
    }

    // Start of a new message: reset reassembly buffer
    if (frag->msg_id != s_reassembly.msg_id || frag->fragment_num == 0) {
        // Only reset when it is genuinely a new message
        if (frag->msg_id != s_reassembly.msg_id) {
            memset(&s_reassembly, 0, sizeof(s_reassembly));
            s_reassembly.msg_id         = frag->msg_id;
            s_reassembly.total_fragments = frag->total_fragments;
        }
    }

    uint8_t fn = frag->fragment_num;

    // Skip duplicate fragments
    if (s_reassembly.received_mask & (1u << fn)) {
        ESP_LOGD(TAG, "Duplicate fragment %d for msg %lu, skipping",
                 fn, (unsigned long)frag->msg_id);
        return;
    }

    // Copy fragment payload into the reassembly buffer
    size_t offset = fn * 1460;
    size_t copy_len = frag->fragment_len;
    if (offset + copy_len > MAX_REASSEMBLY_BUF) {
        ESP_LOGW(TAG, "Fragment overflows reassembly buffer, discarding");
        return;
    }

    memcpy(s_reassembly.buf + offset, frag->data, copy_len);
    s_reassembly.frag_len[fn]     = copy_len;
    s_reassembly.received_mask   |= (1u << fn);

    ESP_LOGD(TAG, "Fragment %d/%d received (%u bytes) for msg %lu from " MACSTR,
             fn + 1, frag->total_fragments, copy_len,
             (unsigned long)frag->msg_id, MAC2STR(recv_info->src_addr));

    // Check if all fragments for this message have arrived
    uint8_t expected_mask = (1u << frag->total_fragments) - 1u;
    if ((s_reassembly.received_mask & expected_mask) == expected_mask) {
        // Compute total reassembled length
        size_t total_len = 0;
        for (int i = 0; i < frag->total_fragments; i++) {
            total_len += s_reassembly.frag_len[i];
        }
        s_reassembly.buf[total_len] = '\0'; // null-terminate

        ESP_LOGI(TAG, "[MSG #%lu] RSSI=%d from " MACSTR " | %d fragment(s) | %zu bytes",
                 (unsigned long)frag->msg_id,
                 recv_info->rx_ctrl->rssi,
                 MAC2STR(recv_info->src_addr),
                 frag->total_fragments,
                 total_len);
        // Print the reassembled CSI data line
        printf("%s\n", s_reassembly.buf);

        // Reset for next message
        memset(&s_reassembly, 0, sizeof(s_reassembly));
    }
}

static void esp_now_init_with_peer(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(esp_now_recv_cb));
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    
    esp_now_rate_config_t rate_config = {
        .phymode = ESP_NOW_PHY_MODE,
        .rate = ESP_NOW_RATE,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));
}

void app_main(void)
{
    // Initialize NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    // Initialize WiFi
    wifi_init();

    // Initialize ESP-NOW with broadcast peer
    esp_now_peer_info_t peer = {
        .channel   = WIFI_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };
    esp_now_init_with_peer(peer);

    // Initialize SD Card (optional)
    if (sd_card_init() == ESP_OK) {
        ESP_LOGI(TAG, "SD card ready for logging");
    } else {
        ESP_LOGW(TAG, "SD card unavailable, continuing without logging");
    }

    ESP_LOGI(TAG, "================ CSI SEND ================");
    ESP_LOGI(TAG, "Channel: %d | Frequency: %d Hz | MAC: " MACSTR,
             WIFI_CHANNEL, SEND_FREQUENCY, MAC2STR(SENDER_MAC));
    ESP_LOGI(TAG, "Listening for ACK messages from receivers...");

    // Main transmission loop
    for (uint32_t count = 0; ; ++count) {
        ret = esp_now_send(peer.peer_addr, (const uint8_t *)&count, sizeof(count));
        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "Send error [%s] - Free heap: %ld bytes", 
                     esp_err_to_name(ret), esp_get_free_heap_size());
        } else {
            ESP_LOGD(TAG, "Sent packet #%lu", (unsigned long)count);
        }

        usleep(1000000 / SEND_FREQUENCY);
    }
}
