from ultralytics import YOLO
import gc
import matplotlib.pyplot as plt

gc.collect()

model = YOLO("yolo11n.pt")

model.train(
    data="/kaggle/working/dataset-aug/data.yaml",
    epochs=20,
    imgsz=288,
    batch=2,
    workers=1,
    pretrained=True,
    optimizer="AdamW"
)

results = model.val()

pr_curve = results.curves_results[0]
x = pr_curve[0].ravel()
y = pr_curve[1].ravel()
plt.figure(figsize=(6,6))
plt.plot(x, y, label="Precision-Recall")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend()
plt.savefig("precision_recall_curve.png")
plt.show()

f1_curve = results.curves_results[1]
x = f1_curve[0].ravel()
y = f1_curve[1].ravel()
plt.figure(figsize=(6,4))
plt.plot(x, y, label="F1 vs Confidence")
plt.xlabel("Confidence")
plt.ylabel("F1-score")
plt.title("F1 vs Confidence")
plt.grid(True)
plt.legend()
plt.savefig("f1_curve.png")
plt.show()

map50_curve = results.curves_results[2]
x = map50_curve[0].ravel()
y = map50_curve[1].ravel()
plt.figure(figsize=(6,4))
plt.plot(x, y, label="mAP50 vs Confidence")
plt.xlabel("Confidence")
plt.ylabel("mAP50")
plt.title("mAP50 vs Confidence")
plt.grid(True)
plt.legend()
plt.savefig("map50_curve.png")
plt.show()