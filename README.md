# GHÉP ẢNH PANORAMA

Ứng dụng web ghép ảnh panorama tự động sử dụng các kỹ thuật xử lý ảnh.
## Nhóm 2
## Thành viên nhóm:
- Phạm Quang Minh - B22DCCN544
- Đoàn Thảo Vân - B22DCCN890
## Thuật Toán Xử Lý Ảnh
- **SIFT (Scale-Invariant Feature Transform)**: Tự cài đặt từ đầu, phát hiện điểm đặc trưng bất biến với scale và rotation
- **Feature Matching**: Ghép cặp đặc trưng giữa các ảnh sử dụng FLANN matcher với k-NN (k=2)
- **Lowe's Ratio Test**: Lọc các cặp ghép tốt với ngưỡng 0.7
- **Homography với RANSAC**: Tính ma trận biến đổi 3x3, loại bỏ outliers (ngưỡng 5.0 pixels)
- **Image Warping & Blending**: Biến đổi phối cảnh và trộn ảnh tạo panorama mượt mà

## Kết quả triển khai
![alt text](/code/app/demo_imgs/image.png)

Ghép 2 ảnh bất kỳ:
![alt text](/code/app/demo_imgs/image-1.png)
![alt text](/code/app/demo_imgs/image-2.png)

Ghép nhiều(6) ảnh cùng lúc:
![alt text](/code/app/demo_imgs/image-3.png)
![alt text](/code/app/demo_imgs/image-4.png)

Ghép ảnh trong đó có 1 ảnh thẳng đúng, một ảnh nằm ngang:
![alt text](/code/app/demo_imgs/image-5.png)
![alt text](/code/app/demo_imgs/image-6.png)

## 📄 License

MIT License - Tự do sử dụng, chỉnh sửa và phân phối.

---

