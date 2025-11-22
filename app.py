from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS, cross_origin
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import uuid
import time

app = Flask(__name__)
# Bật CORS để cho phép gọi từ frontend (127.0.0.1:5500, live server, v.v.)
# Có thể giới hạn origin cụ thể nếu muốn an toàn hơn
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5500", "http://localhost:5500"]}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULT_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Tạo thư mục nếu chưa tồn tại
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class PanoramaStitcher:
    """Class xử lý ghép ảnh panorama sử dụng SIFT, Feature Matching, Homography, RANSAC"""
    
    def __init__(self):
        # VIETLAI: Viết lại thuật toán SIFT 
        from sift import SIFT
        self.sift = SIFT(
            n_octave_layers=3,
            contrast_threshold=0.04,
            edge_threshold=10,
            sigma=1.6
        )
        
    def detect_and_compute(self, image):
        """
        Phát hiện keypoints và tính toán descriptors sử dụng SIFT
        
        SIFT gồm 2 bước chính:
        1. Keypoint Detection:
           - Xây dựng Gaussian Pyramid (nhiều octaves, mỗi octave nhiều scales)
           - Tính Difference of Gaussians (DoG) 
           - Tìm local extrema trong không gian 3D (x, y, scale)
           - Lọc keypoints yếu bằng contrast và Harris response
        
        2. Descriptor Generation:
           - Xác định hướng chính (orientation) cho mỗi keypoint
           - Trích vùng 16×16 xung quanh keypoint, xoay theo hướng chính
           - Chia thành 16 subregions (4×4)
           - Mỗi subregion: histogram 8 bins của gradient orientation
           - Vector 128 chiều (16 × 8), chuẩn hóa L2-norm
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # VIETLAI
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        return keypoints, descriptors
    
    def match_features(self, desc1, desc2):
        """
        Ghép cặp đặc trưng sử dụng FLANN matcher và Lowe's ratio test
        
        FLANN (Fast Library for Approximate Nearest Neighbors):
        - Sử dụng KD-tree để tìm kiếm nhanh trong không gian 128 chiều
        - Tìm k=2 nearest neighbors cho mỗi descriptor
        
        Lowe's Ratio Test:
        - So sánh khoảng cách của match tốt nhất (m) với match tốt thứ 2 (n)
        - Nếu m.distance < 0.7 * n.distance → match tốt (inlier potential)
        - Ngưỡng 0.7 giúp loại bỏ các matches mơ hồ
        """
        # FLANN parameters cho SIFT (descriptor float 128-dim)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Số lần kiểm tra, cao hơn = chính xác hơn nhưng chậm hơn
        
        # VIETLAI: Thay FLANN bằng matcher tự cài đặt (module hóa)
        # - Sử dụng khoảng cách Euclid giữa các descriptor
        # - Trả về 2-NN cho mỗi descriptor bên trái để dùng Lowe's ratio test
        from matcher import knn_match
        matches = knn_match(desc1, desc2, k=2)
        
        # Lowe's ratio test để lọc matches tốt
        good_matches = []
        for match_pair in matches:
            # Đảm bảo có đủ 2 matches
            if len(match_pair) == 2:
                m, n = match_pair
                # Chỉ giữ matches có ratio < 0.7
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def find_homography_ransac(self, kp1, kp2, matches):
        """
        Tìm ma trận Homography sử dụng RANSAC
        
        RANSAC (Random Sample Consensus):
        1. Chọn ngẫu nhiên 4 cặp điểm từ matches
        2. Tính homography H (ma trận 3×3) từ 4 cặp điểm này:
           - Giải hệ phương trình Ah = b (8 phương trình, 8 ẩn)
           - Mỗi cặp điểm cho 2 phương trình
        3. Tính reprojection error cho tất cả các điểm:
           error = ||p'ᵢ - H·pᵢ||
        4. Đếm số inliers (error < threshold)
        5. Lặp lại nhiều lần, chọn H có nhiều inliers nhất
        6. Dùng least squares với tất cả inliers để tính H tối ưu cuối cùng
        
        OpenCV's findHomography đã implement RANSAC tối ưu
        """
        if len(matches) < 4:
            return None, None
        
        # Lấy tọa độ các điểm tương ứng
        # src_pts: điểm trên ảnh mới (images[i])
        # dst_pts: điểm trên ảnh panorama hiện tại
        src_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Tìm homography với RANSAC
        # method=cv2.RANSAC: sử dụng thuật toán RANSAC
        # ransacReprojThreshold=5.0: ngưỡng error để coi là inlier (pixels)
        # VIETLAI: Tự cài đặt RANSAC + DLT (module hóa, không dùng cv2.findHomography)
        from homography import find_homography_ransac
        H, mask = find_homography_ransac(src_pts, dst_pts, ransac_reproj_threshold=5.0, max_iters=2000)
        
        # mask: array [n×1] với 1=inlier, 0=outlier
        inliers_count = np.sum(mask) if mask is not None else 0
        
        return H, mask
    
    def warp_and_blend(self, img1, img2, H):
        """
        Biến đổi và trộn ảnh sử dụng Perspective Transform và Alpha Blending
        
        Perspective Transform (Warping):
        - Áp dụng ma trận homography H lên img2
        - Công thức: [x', y', w'] = H · [x, y, 1]
        - Tọa độ thực tế: (x'/w', y'/w')
        
        Alpha Blending:
        - Trong vùng chồng lấn, trộn 2 ảnh mượt mà
        - Sử dụng gradient weights để tránh đường nối rõ ràng
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Tìm kích thước canvas cho panorama
        # Transform 4 góc của img2 để biết vùng nó chiếm sau khi warp
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        #VIETLAI
        corners2_transformed = cv2.perspectiveTransform(corners2, H)
        all_corners = np.concatenate((corners1, corners2_transformed), axis=0)
        
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Ma trận dịch chuyển để đảm bảo tất cả pixels nằm trong canvas dương
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        
        #VIETLAI
        # Warp img2 với homography + translation
        output_shape = (x_max - x_min, y_max - y_min)
        panorama = cv2.warpPerspective(img2, translation.dot(H), output_shape)
        
        # Vị trí đặt img1 vào panorama
        y_start = -y_min
        y_end = h1 - y_min
        x_start = -x_min
        x_end = w1 - x_min
        
        # Đảm bảo không vượt quá kích thước panorama
        y_start = max(0, y_start)
        y_end = min(panorama.shape[0], y_end)
        x_start = max(0, x_start)
        x_end = min(panorama.shape[1], x_end)
        
        # Tính offset tương ứng trong img1
        img1_y_start = max(0, -(-y_min))
        img1_y_end = img1_y_start + (y_end - y_start)
        img1_x_start = max(0, -(-x_min))
        img1_x_end = img1_x_start + (x_end - x_start)
        
        # Lấy vùng tương ứng từ img1
        img1_region = img1[img1_y_start:img1_y_end, img1_x_start:img1_x_end]
        pano_region = panorama[y_start:y_end, x_start:x_end]
        
        # Tạo mask cho vùng overlap
        # Vùng nào của img2 (đã warp) có nội dung (không đen)
        gray_pano = cv2.cvtColor(pano_region, cv2.COLOR_BGR2GRAY)
        mask2_region = (gray_pano > 1).astype(np.uint8)
        
        # Vùng overlap là nơi cả 2 ảnh đều có nội dung
        mask1_region = np.ones(img1_region.shape[:2], dtype=np.uint8)
        overlap_mask = cv2.bitwise_and(mask1_region, mask2_region)
        
        # Alpha blending trong vùng overlap
        if np.sum(overlap_mask) > 0:
            # Tìm cột bắt đầu và kết thúc của overlap
            overlap_cols = np.where(np.sum(overlap_mask, axis=0) > 0)[0]
            
            if len(overlap_cols) > 0:
                col_start = overlap_cols[0]
                col_end = overlap_cols[-1]
                blend_width = col_end - col_start + 1
                
                # Tạo alpha map (gradient từ 0 đến 1)
                for col in range(col_start, col_end + 1):
                    alpha = (col - col_start) / max(1, blend_width)
                    
                    for row in range(overlap_mask.shape[0]):
                        if overlap_mask[row, col]:
                            # Blend pixel
                            pano_region[row, col] = (
                                (1 - alpha) * img1_region[row, col].astype(float) +
                                alpha * pano_region[row, col].astype(float)
                            ).astype(np.uint8)
        
        # Vùng không overlap: dùng img1
        non_overlap_mask = (mask2_region == 0)
        pano_region[non_overlap_mask] = img1_region[non_overlap_mask]
        
        # Ghi lại vào panorama
        panorama[y_start:y_end, x_start:x_end] = pano_region
        
        return panorama
    
    def crop_black_borders(self, img):
        """
        Cắt bỏ viền đen xung quanh panorama
        
        Sau khi warp, thường có vùng đen (pixel = 0) ở viền do perspective transform
        Hàm này tìm bounding box của vùng có nội dung và crop lại
        """
        # Chuyển sang grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Tìm vùng không đen (threshold > 1 để tránh noise)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        
        # Tìm contours của vùng có nội dung
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) > 0:
            # Lấy bounding box lớn nhất
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            
            # Thêm padding nhỏ để không crop sát quá
            padding = 2
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(img.shape[1] - x, w + 2 * padding)
            h = min(img.shape[0] - y, h + 2 * padding)
            
            return img[y:y+h, x:x+w]
        
        return img
    
    def stitch(self, images):
        """
        Ghép nhiều ảnh thành panorama
        
        Pipeline:
        1. Bắt đầu với ảnh đầu tiên làm base
        2. Với mỗi ảnh tiếp theo:
           a. SIFT: Detect keypoints + compute descriptors
           b. Feature Matching: FLANN + Lowe's ratio test
           c. RANSAC: Tìm homography H và loại bỏ outliers
           d. Perspective Transform: Warp ảnh mới vào panorama
           e. Alpha Blending: Trộn vùng chồng lấn
        3. Crop viền đen
        """
        if len(images) < 2:
            return None
        
        # Bắt đầu với ảnh đầu tiên
        result = images[0]
        
        print(f"\n{'='*60}")
        print(f"BẮT ĐẦU GHÉP {len(images)} ẢNH THÀNH PANORAMA")
        print(f"{'='*60}\n")
        
        # Lần lượt ghép các ảnh tiếp theo
        for i in range(1, len(images)):
            print(f"[Bước {i}/{len(images)-1}] Ghép ảnh {i+1} vào panorama...")
            
            # 1. SIFT: Phát hiện keypoints và descriptors
            print("  → SIFT: Detecting keypoints & computing descriptors...")
            kp1, desc1 = self.detect_and_compute(result)
            kp2, desc2 = self.detect_and_compute(images[i])
            print(f"     ✓ Panorama hiện tại: {len(kp1)} keypoints")
            print(f"     ✓ Ảnh {i+1}: {len(kp2)} keypoints")
            
            # 2. Feature Matching với FLANN + Lowe's ratio test
            print("  → Feature Matching: FLANN + Lowe's ratio test (threshold=0.7)...")
            matches = self.match_features(desc1, desc2)
            print(f"     ✓ Tìm thấy {len(matches)} cặp matches tốt")
            
            if len(matches) < 10:
                print(f"     ✗ CẢNH BÁO: Không đủ matches (< 10), bỏ qua ảnh {i+1}\n")
                continue
            
            # 3. RANSAC: Tìm homography và loại outliers
            print("  → RANSAC: Tìm homography matrix & loại bỏ outliers...")
            H, mask = self.find_homography_ransac(kp1, kp2, matches)
            
            if H is None:
                print(f"     ✗ CẢNH BÁO: Không tìm thấy homography, bỏ qua ảnh {i+1}\n")
                continue
            
            inliers = np.sum(mask) if mask is not None else 0
            outliers = len(matches) - inliers
            inlier_ratio = (inliers / len(matches)) * 100
            print(f"     ✓ Inliers: {inliers}/{len(matches)} ({inlier_ratio:.1f}%)")
            print(f"     ✓ Outliers: {outliers} (đã loại bỏ)")
            
            # 4. Warp & Blend
            print("  → Perspective Transform + Alpha Blending...")
            result = self.warp_and_blend(result, images[i], H)
            print(f"     ✓ Đã ghép thành công ảnh {i+1}")
            print(f"     ✓ Kích thước panorama: {result.shape[1]}×{result.shape[0]} pixels\n")
        
        # 5. Cắt bỏ viền đen
        print("→ Cropping black borders...")
        result = self.crop_black_borders(result)
        print(f"  ✓ Kích thước cuối cùng: {result.shape[1]}×{result.shape[0]} pixels")
        
        print(f"\n{'='*60}")
        print(f"✓ HOÀN THÀNH PANORAMA!")
        print(f"{'='*60}\n")
        
        return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stitch', methods=['POST', 'OPTIONS'])
@cross_origin(origins=["http://127.0.0.1:5500", "http://localhost:5500"])
def stitch_images():
    try:
        # Kiểm tra có file không
        if 'images' not in request.files:
            return jsonify({'error': 'Không có file nào được tải lên'}), 400
        
        files = request.files.getlist('images')
        
        if len(files) < 2:
            return jsonify({'error': 'Cần ít nhất 2 ảnh để ghép panorama'}), 400
        
        # Đọc và lưu các ảnh
        images = []
        for file in files:
            if file and allowed_file(file.filename):
                # Đọc ảnh từ file
                file_bytes = np.frombuffer(file.read(), np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Resize ảnh nếu quá lớn
                    max_dim = 800
                    h, w = img.shape[:2]
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        img = cv2.resize(img, None, fx=scale, fy=scale)
                    
                    images.append(img)
        
        if len(images) < 2:
            return jsonify({'error': 'Không đủ ảnh hợp lệ để ghép'}), 400
        
        # Ghép ảnh panorama
        stitcher = PanoramaStitcher()
        panorama = stitcher.stitch(images)
        
        if panorama is None:
            return jsonify({'error': 'Không thể ghép ảnh. Vui lòng chọn các ảnh có phần chồng lấn'}), 400
        
        # Lưu kết quả
        result_id = str(uuid.uuid4())
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'{result_id}.jpg')
        cv2.imwrite(result_path, panorama)
        
        return jsonify({
            'success': True,
            'result_id': result_id,
            'message': f'Đã ghép thành công {len(images)} ảnh!'
        })
        
    except Exception as e:
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500

@app.route('/result/<result_id>')
def get_result(result_id):
    try:
        result_path = os.path.join(app.config['RESULT_FOLDER'], f'{result_id}.jpg')
        if os.path.exists(result_path):
            return send_file(result_path, mimetype='image/jpeg')
        else:
            return jsonify({'error': 'Không tìm thấy ảnh'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
