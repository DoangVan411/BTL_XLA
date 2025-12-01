# GHÉP ẢNH PANORAMA

Ứng dụng web ghép ảnh panorama tự động sử dụng các kỹ thuật xử lý ảnh nâng cao. Hỗ trợ giao diện web Flask

### Link Slide trình bày báo cáo
[Slide trình bày BTL](https://www.canva.com/design/DAG4mOlDkIQ/-DEv2SrYDvIprM9vEf9RVQ/edit?utm_content=DAG4mOlDkIQ&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## ✨ Tính Năng Chính

### Thuật Toán Xử Lý Ảnh
- **SIFT (Scale-Invariant Feature Transform)**: Tự cài đặt từ đầu, phát hiện điểm đặc trưng bất biến với scale và rotation
- **Feature Matching**: Ghép cặp đặc trưng giữa các ảnh sử dụng FLANN matcher với k-NN (k=2)
- **Lowe's Ratio Test**: Lọc các cặp ghép tốt với ngưỡng 0.7
- **Homography với RANSAC**: Tính ma trận biến đổi 3x3, loại bỏ outliers (ngưỡng 5.0 pixels)
- **Image Warping & Blending**: Biến đổi phối cảnh và trộn ảnh tạo panorama mượt mà

### Giao Diện
- **Flask Web App**: Giao diện web đơn giản với HTML/CSS/JavaScript
- **REST API**: Endpoint để tích hợp vào ứng dụng khác

## 📋 Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên (khuyến nghị 3.10+)
- **OpenCV**: opencv-python và opencv-contrib-python (hỗ trợ SIFT)
- **Flask**: Framework web chính
- **NumPy**: Xử lý mảng và ma trận

## 🔧 Cài Đặt

1. **Clone repository**
```bash
git clone <repository-url>
cd XLA
```

2. **Cài đặt dependencies**
```bash
pip install -r requirements.txt
```

## 🎮 Hướng Dẫn Sử Dụng

1. **Khởi động server**
```bash
python -m app.app
```

2. **Truy cập ứng dụng**: Trình duyệt sẽ tự động mở `http://localhost:5000`

3. **Upload và ghép ảnh**:
   - Click vào khu vực upload hoặc kéo thả 2+ ảnh
   - Định dạng hỗ trợ: JPG, JPEG, PNG
   - Kích thước tối đa: 16MB/ảnh
   - Click "Ghép Ảnh Panorama"
   - Tải xuống kết quả

### API Endpoint

```bash
POST /api/stitch
Content-Type: multipart/form-data

# Gửi file ảnh với key "images[]"
# Response: JSON với ảnh panorama dạng base64
```


## 🛠️ Cấu Trúc Dự Án

```
XLA/
├── app/
│   ├── __init__.py
│   ├── app.py                 # Entry point chạy Flask app
│   ├── factory.py             # Flask app factory pattern
│   ├── config.py              # Cấu hình (port, paths, limits)
│   ├── streamlit_app.py       # Giao diện Streamlit
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes.py          # Flask Blueprint (API endpoints)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   └── panorama_service.py # Logic ghép ảnh chính
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_io.py        # Đọc/ghi/encode ảnh
│   │   └── paths.py           # Xử lý đường dẫn
│   │
│   ├── sift.py                # SIFT implementation
│   ├── matcher.py             # Feature matching + Lowe's test
│   ├── homography.py          # Homography + RANSAC
│   └── transform.py           # Warping & blending
│
├── templates/
│   └── index.html             # Giao diện web Flask
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
│
├── nature/                    # Ảnh mẫu test (nếu có)
├── uploads/                   # Thư mục tạm (gitignored)
│
├── requirements.txt           # Python dependencies
└── README.md
```

## Kết quả triển khai
![alt text](/app/demo_imgs/image.png)

Ghép 2 ảnh bất kỳ:
![alt text](/app/demo_imgs/image-1.png)
![alt text](/app/demo_imgs/image-2.png)

Ghép nhiều(6) ảnh cùng lúc:
![alt text](/app/demo_imgs/image-3.png)
![alt text](/app/demo_imgs/image-4.png)

Ghép ảnh trong đó có 1 ảnh thẳng đúng, một ảnh nằm ngang:
![alt text](/app/demo_imgs/image-5.png)
![alt text](/app/demo_imgs/image-6.png)

## 📄 License

MIT License - Tự do sử dụng, chỉnh sửa và phân phối.

---

