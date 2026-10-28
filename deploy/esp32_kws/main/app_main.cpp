// Milestone 1 (docs/ONCHIP.md): run int8 BC-ResNet-8 on the ESP32 over the four
// baked-in log-mel test vectors and check the argmax against the host.
#include <cmath>
#include <cstring>

#include "dl_model_base.hpp"
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "test_vectors.h"

extern const uint8_t bcresnet8_espdl[] asm("_binary_bcresnet8_espdl_start");
static const char *TAG = "kws";

static void log_heap(const char *when)
{
    ESP_LOGI(TAG, "[heap %s] internal free %u B, largest block %u B, min ever free %u B", when,
             (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL),
             (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL));
}

extern "C" void app_main(void)
{
    log_heap("boot");

    // param_copy=false: weights stay in flash; the ESP32 has no room to copy ~321 KB into DRAM.
    dl::Model *model = new dl::Model((const char *)bcresnet8_espdl, fbs::MODEL_LOCATION_IN_FLASH_RODATA, 0,
                                     dl::MEMORY_MANAGER_GREEDY, nullptr, false);
    log_heap("after load");

    dl::TensorBase *in = model->get_inputs().begin()->second;
    dl::TensorBase *out = model->get_outputs().begin()->second;
    if (!in || !out || !in->data || !out->data) {
        // The activation arena could not be allocated (esp-dl logs the size it wanted above).
        ESP_LOGE(TAG, "model tensors unallocated -- activation arena did not fit in internal RAM; stopping");
        return;
    }
    auto ishape = in->get_shape();
    ESP_LOGI(TAG, "input '%s' %s shape [%d,%d,%d,%d] exp %d | output '%s' %s size %d exp %d",
             model->get_inputs().begin()->first.c_str(), in->get_dtype_string(), ishape[0], ishape[1], ishape[2],
             ishape[3], in->get_exponent(), model->get_outputs().begin()->first.c_str(), out->get_dtype_string(),
             out->get_size(), out->get_exponent());

    // Bit-exactness against esp-ppq's simulation on the embedded test input ("yes").
    esp_err_t t = model->test();
    ESP_LOGI(TAG, "model->test() vs esp-ppq: %s", t == ESP_OK ? "PASS" : "FAIL");

    if (in->get_size() != TV_MELS * TV_FRAMES || out->get_size() != TV_CLASSES) {
        ESP_LOGE(TAG, "unexpected tensor sizes: in %d out %d", in->get_size(), out->get_size());
        return;
    }

    const float in_scale = DL_RESCALE(in->get_exponent());
    const float out_scale = DL_SCALE(out->get_exponent());
    int passed = 0;
    int64_t total_us = 0;

    for (int v = 0; v < TV_COUNT; v++) {
        // Input is C=1, so NCHW (1,1,40,101) and ESP-DL's NHWC (1,40,101,1) share the
        // same mel-major memory order: copy straight across.
        int8_t *dst = (int8_t *)in->data;
        for (int i = 0; i < TV_MELS * TV_FRAMES; i++) {
            dst[i] = dl::quantize<int8_t>(tv_input[v][i], in_scale);
        }

        int64_t t0 = esp_timer_get_time();
        model->run();
        int64_t dt = esp_timer_get_time() - t0;
        total_us += dt;

        const int8_t *q = (const int8_t *)out->data;
        int best = 0, second = -1;
        for (int c = 1; c < TV_CLASSES; c++) {
            if (q[c] > q[best]) {
                second = best;
                best = c;
            } else if (second < 0 || q[c] > q[second]) {
                second = c;
            }
        }
        float margin = (q[best] - q[second]) * out_scale;
        bool ok = best == tv_expected_label[v];
        passed += ok;

        printf("\nTV%d expected %d (%s) -> predicted %d (%s) %s | margin %.3f | %.2f ms\n", v, tv_expected_label[v],
               tv_class_names[tv_expected_label[v]], best, tv_class_names[best], ok ? "MATCH" : "MISMATCH", margin,
               dt / 1000.0);
        printf("  chip : ");
        for (int c = 0; c < TV_CLASSES; c++) printf("%8.3f", q[c] * out_scale);
        printf("\n  host : ");
        for (int c = 0; c < TV_CLASSES; c++) printf("%8.3f", tv_host_logits[v][c]);
        printf("\n");
    }

    printf("\nRESULT: %d/%d predictions match tv_expected_label; mean inference %.2f ms\n", passed, TV_COUNT,
           total_us / 1000.0 / TV_COUNT);
    log_heap("after inference");
    model->profile_memory();
    model->profile_module(true);
}
