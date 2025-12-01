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