from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

# Cấu hình mặc định cho lần dự đoán này.
MODEL_PATH = BASE_DIR / "best_char_obb.pt"
SOURCE_IMAGE = BASE_DIR / "qq1.png"
OUTPUT_DIR = BASE_DIR / "predict_results"
CONFIDENCE = 0.25
IMAGE_SIZE = 1024


def main() -> None:
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {MODEL_PATH}")
    if not SOURCE_IMAGE.is_file():
        raise FileNotFoundError(f"Không tìm thấy ảnh: {SOURCE_IMAGE}")

    # Ultralytics sẽ tạo OUTPUT_DIR nếu thư mục chưa tồn tại.
    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(SOURCE_IMAGE),
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        save=True,
        project=str(OUTPUT_DIR),
        name="qq1",
        exist_ok=True,
    )

    result_dir = Path(results[0].save_dir).resolve()
    print(f"Đã lưu kết quả tại: {result_dir}")


if __name__ == "__main__":
    main()
