import sensor
import time
import ml
import gc

print("Free heap before model load:", gc.mem_free())

sensor.reset()
sensor.set_pixformat(sensor.RGB565)   # model was trained on RGB
sensor.set_framesize(sensor.QVGA)     # 320x240, then windowed to 96x96
sensor.set_windowing((96, 96))        # center-crop to match model input
sensor.skip_frames(time=2000)

try:
    model = ml.Model("custom_objects_int8_vela.tflite", load_to_fb=True)
except Exception as e:
    raise Exception('Failed to load model. Copy custom_objects_int8_vela.tflite to the board. (' + str(e) + ')')

print("Input shape:", model.input_shape)
print("Input dtype:", model.input_dtype)
print("Output shape:", model.output_shape)

try:
    labels = [line.rstrip('\n') for line in open("custom_objects_labels.txt")]
except Exception as e:
    raise Exception('Failed to load labels. Copy custom_objects_labels.txt to the board. (' + str(e) + ')')

CONFIDENCE_THRESHOLD = 0.6

clock = time.clock()
while True:
    clock.tick()
    img = sensor.snapshot()

    outputs = model.predict([img])
    scores = outputs[0].flatten().tolist()

    top_idx = scores.index(max(scores))
    top_label = labels[top_idx]
    top_score = scores[top_idx]

    # int8 models output raw logits; convert via softmax approximation isn't needed —
    # the largest value is the predicted class regardless of scale
    if top_score > CONFIDENCE_THRESHOLD:
        color = (0, 255, 0)     # green when confident
    else:
        color = (255, 100, 0)   # orange when uncertain

    img.draw_string(4, 4, "%s" % top_label, color=color, scale=2, mono_space=False)
    img.draw_string(4, 24, "%.2f" % top_score, color=color, scale=2, mono_space=False)

    print("%s  %.2f  %.1f fps" % (top_label, top_score, clock.fps()))
