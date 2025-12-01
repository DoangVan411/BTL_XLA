# Panorama Image Stitcher

Ứng dụng web ghép ảnh panorama sử dụng các kỹ thuật xử lý ảnh nâng cao.

## 🚀 Tính Năng

- **SIFT (Scale-Invariant Feature Transform)**: Phát hiện điểm đặc trưng trong ảnh
- **Feature Matching**: Ghép cặp đặc trưng giữa các ảnh sử dụng FLANN matcher
- **Lowe's Ratio Test**: Lọc các cặp ghép tốt với ngưỡng 0.7
- **Homography**: Tính ma trận biến đổi 3x3 giữa các ảnh
- **RANSAC**: Loại bỏ outliers và tìm homography chính xác
- **Image Warping & Blending**: Biến đổi và trộn ảnh tạo panorama mượt mà

## 📋 Yêu Cầu

- Python 3.8+
- OpenCV với module contrib (SIFT)
- Flask

## 🔧 Cài Đặt

1. Clone repository hoặc tải về mã nguồn

2. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements.txt
```

## 🎮 Sử Dụng

1. Chạy ứng dụng (local, không Docker):
```bash
python -m app.app
```

2. Mở trình duyệt và truy cập: `http://localhost:5000`

3. Upload 2 hoặc nhiều ảnh có phần chồng lấn

4. Click "Ghép Ảnh Panorama" để tạo ảnh toàn cảnh

5. Tải xuống kết quả

## 🐳 Chạy bằng Docker

1. Build image:
```bash
docker build -t panorama-stitcher .
```

2. Chạy container (Linux/macOS):
```bash
docker run --rm -p 5000:5000 panorama-stitcher
```

2. Chạy container (Windows CMD):
```bash
docker run --rm -p 5000:5000 panorama-stitcher
```

3. Truy cập: `http://localhost:5000`

## 📝 Lưu Ý

- Các ảnh nên có phần chồng lấn ít nhất 30-40%
- Chụp ảnh từ cùng một vị trí, xoay camera theo chiều ngang
- Tránh các vật thể di chuyển trong khung hình
- Độ phân giải ảnh sẽ được tự động điều chỉnh để tối ưu hiệu suất
- Ứng dụng trả ảnh kết quả dưới dạng base64 qua API/HTTP response

## 🛠️ Cấu Trúc Dự Án

```
BTL_XLA/
├── app/
│   ├── app.py             # Entry chạy Flask (python -m app.app)
│   ├── factory.py         # Khởi tạo Flask app
│   ├── api/
│   │   └── routes.py      # HTTP routes (Blueprint)
│   ├── services/
│   │   └── panorama_service.py
│   ├── utils/
│   │   └── image_io.py
│   ├── sift.py, matcher.py, homography.py, transform.py
│   └── config.py
├── requirements.txt       # Thư viện Python
├── Dockerfile             # Đóng gói/chạy bằng Docker
├── templates/
│   └── index.html        # Giao diện web

```

## 🎯 Các Kỹ Thuật Xử Lý Ảnh

### 1. SIFT (Scale-Invariant Feature Transform)
- Phát hiện keypoints bất biến với scale và rotation
- Tạo descriptors 128 chiều cho mỗi keypoint

### 2. Feature Matching
- Sử dụng FLANN (Fast Library for Approximate Nearest Neighbors)
- K-NN matching với k=2

### 3. Lowe's Ratio Test
- Lọc matches tốt với điều kiện: distance(m) < 0.7 * distance(n)

### 4. Homography với RANSAC
- Tìm ma trận biến đổi 3x3 
- RANSAC loại bỏ outliers với ngưỡng 5.0 pixels

### 5. Warping & Blending
- Perspective transform sử dụng homography
- Tính toán canvas size phù hợp
- Trộn ảnh tự nhiên

## 📄 License

MIT License
