#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT20  // 20 MHz = 256 subcarriers for 802.11ax
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11AX

#define CONFIG_ESP_NOW_PHYMODE              WIFI_PHY_MODE_HE20  // 802.11ax HE SU mode
#define CONFIG_ESP_NOW_RATE                 WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   0
#define CONFIG_GAIN_CONTROL                 1

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";

typedef struct {
    uint32_t msg_id;
    uint8_t fragment_num;    // Nomor fragment saat ini (0, 1, 2, ...)
    uint8_t total_fragments; // Total jumlah fragment
    uint16_t fragment_len;   // Panjang data dalam fragment ini
    char data[1460];         // Data CSI fragment (1470 - 10 bytes header)
} csi_fragment_msg_t;

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));

    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
    if (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }

    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,//  WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));

}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf) {
        ESP_LOGW(TAG, "<%s> wifi_csi_cb", esp_err_to_name(ESP_ERR_INVALID_ARG));
        return;
    }

    if (memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;
    }

    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_count = 0;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;
    
    // Buffer untuk menyimpan data CSI lengkap
    static char csi_data_string[2048]; // Buffer cukup besar untuk menampung data CSI
    
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (s_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    } else if (s_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
        ESP_LOGD(TAG, "fft_force %d, agc_force %d", fft_gain_baseline, agc_gain_baseline);
#endif
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
    
    ESP_LOGI(TAG, "[PKT #%d] | Rate: %d | RSSI: %d dBm | Gain - compensate: %.2f, agc: %d, fft: %d", 
             s_count, rx_ctrl->rate, rx_ctrl->rssi, compensate_gain, agc_gain, fft_gain);

    uint32_t rx_id = *(uint32_t *)(info->payload + 15);
    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_state,len,first_word,data\n");
    }

    // Simpan data CSI ke dalam string terlebih dahulu
    int string_offset = 0;
    string_offset += snprintf(csi_data_string + string_offset, sizeof(csi_data_string) - string_offset,
                             "CSI_DATA,%lu," MACSTR ",%d,%d,%d,%d,%d,%d,%lu,%d,%d,%d,%d,\"[%d",
                             (unsigned long)rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate,
                             rx_ctrl->noise_floor, fft_gain, agc_gain, rx_ctrl->channel,
                             (unsigned long)rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->rx_state,
                             info->len, info->first_word_invalid, (int16_t)(compensate_gain * info->buf[0]));
    
    for (int i = 1; i < info->len && string_offset < (int)sizeof(csi_data_string) - 10; i++) {
        string_offset += snprintf(csi_data_string + string_offset, sizeof(csi_data_string) - string_offset,
                                 ",%d", (int16_t)(compensate_gain * info->buf[i]));
    }
    
    if (string_offset < (int)sizeof(csi_data_string) - 3) {
        string_offset += snprintf(csi_data_string + string_offset, sizeof(csi_data_string) - string_offset, "]\"");
    }
    
    // Log data CSI menggunakan string yang sudah disimpan
    ets_printf("%s\n", csi_data_string);
    
    // Kirim data CSI lengkap menggunakan ESP NOW v2.0 secara dinamis
    // ESP NOW v2.0 maksimal 1470 bytes (ESP_NOW_MAX_DATA_LEN_V2)
    size_t data_len = strlen(csi_data_string);
    const size_t max_fragment_data = 1460; // 1470 - 10 bytes header
    uint8_t total_fragments = (data_len + max_fragment_data - 1) / max_fragment_data; // Ceiling division
    
    bool peer_exists = esp_now_is_peer_exist(info->mac);
    if (!peer_exists) {
        esp_now_peer_info_t sender_peer = {
            .channel = rx_ctrl->channel,
            .ifidx = WIFI_IF_STA,
            .encrypt = false,
        };
        memcpy(sender_peer.peer_addr, info->mac, 6);
        
        esp_err_t add_result = esp_now_add_peer(&sender_peer);
        if (add_result == ESP_OK) {
            esp_now_rate_config_t rate_config = {
                .phymode = CONFIG_ESP_NOW_PHYMODE,
                .rate = CONFIG_ESP_NOW_RATE,
                .ersu = false,
                .dcm = false
            };
            esp_now_set_peer_rate_config(sender_peer.peer_addr, &rate_config);
        }
    }
    
    // Kirim data dalam fragments
    for (uint8_t frag_num = 0; frag_num < total_fragments; frag_num++) {
        csi_fragment_msg_t fragment_msg = {0};
        fragment_msg.msg_id = rx_id;
        fragment_msg.fragment_num = frag_num;
        fragment_msg.total_fragments = total_fragments;
        
        size_t offset = frag_num * max_fragment_data;
        size_t remaining_data = data_len - offset;
        size_t current_fragment_len = (remaining_data > max_fragment_data) ? max_fragment_data : remaining_data;
        
        fragment_msg.fragment_len = current_fragment_len;
        memcpy(fragment_msg.data, csi_data_string + offset, current_fragment_len);
        
        size_t packet_size = sizeof(uint32_t) + sizeof(uint8_t) * 2 + sizeof(uint16_t) + current_fragment_len;
        
        esp_err_t send_result = esp_now_send(info->mac, (uint8_t*)&fragment_msg, packet_size);
        if (send_result != ESP_OK) {
            ESP_LOGW(TAG, "Failed to send CSI fragment %d/%d for packet %lu: %s", 
                     frag_num + 1, total_fragments, (unsigned long)rx_id, esp_err_to_name(send_result));
        } else {
            ESP_LOGD(TAG, "CSI fragment %d/%d sent (%zu bytes) for packet %lu to " MACSTR, 
                     frag_num + 1, total_fragments, current_fragment_len, (unsigned long)rx_id, MAC2STR(info->mac));
        }
        
        // Delay kecil antar fragment untuk mencegah buffer overflow di receiver
        vTaskDelay(pdMS_TO_TICKS(5));
    }
    
    ESP_LOGI(TAG, "All CSI fragments sent (%d fragments, %zu total bytes) for packet %lu", 
             total_fragments, data_len, (unsigned long)rx_id);
    
    s_count++;
}

static void wifi_csi_init()
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = false,  // Disable 802.11b/g
        .acquire_csi_ht20       = false,  // Disable 802.11n 20MHz
        .acquire_csi_ht40       = false,  // Disable 802.11n 40MHz
        .acquire_csi_su         = true,   // Enable 802.11ax HE SU (256 subcarriers @ 20MHz)
        .acquire_csi_mu         = false,  // Disable MU-MIMO for now
        .acquire_csi_dcm        = true,   // Enable Dual Carrier Modulation
        .acquire_csi_beamformed = true,   // Enable beamformed packets
        .acquire_csi_he_stbc    = 2,      // HE STBC mode
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main()
{
    /**
     * @brief Initialize NVS
     */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /**
     * @brief Initialize Wi-Fi
     */
    wifi_init();

    /**
     * @brief Initialize ESP-NOW
     *        ESP-NOW protocol see: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html
     */

    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };

    wifi_esp_now_init(peer);

    wifi_csi_init();
}
