let selectedFiles = [];

const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const previewSection = document.getElementById('previewSection');
const previewGrid = document.getElementById('previewGrid');
const imageCount = document.getElementById('imageCount');
const stitchBtn = document.getElementById('stitchBtn');
const loading = document.getElementById('loading');
const message = document.getElementById('message');
const resultSection = document.getElementById('resultSection');
const resultImage = document.getElementById('resultImage');
const downloadBtn = document.getElementById('downloadBtn');

// Click để chọn file
uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// Xử lý file được chọn
fileInput.addEventListener('change', (e) => {
    handleFiles(e.target.files);
});

// Drag and drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});

function handleFiles(files) {
    const validFiles = Array.from(files).filter(file =>
        file.type === 'image/jpeg' || file.type === 'image/png'
    );

    selectedFiles = [...selectedFiles, ...validFiles];
    updatePreview();
}

function updatePreview() {
    previewGrid.innerHTML = '';
    imageCount.textContent = selectedFiles.length;

    if (selectedFiles.length > 0) {
        previewSection.style.display = 'block';

        selectedFiles.forEach((file, index) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                const div = document.createElement('div');
                div.className = 'preview-item';
                div.innerHTML = `
                    <img src="${e.target.result}" alt="Preview ${index + 1}">
                    <button class="remove-btn" onclick="removeImage(${index})">×</button>
                `;
                previewGrid.appendChild(div);
            };
            reader.readAsDataURL(file);
        });

        stitchBtn.disabled = selectedFiles.length < 2;
    } else {
        previewSection.style.display = 'none';
    }

    hideMessage();
    resultSection.classList.remove('show');
}

function removeImage(index) {
    selectedFiles.splice(index, 1);
    updatePreview();
}

// Ghép ảnh
stitchBtn.addEventListener('click', async () => {
    if (selectedFiles.length < 2) {
        showMessage('Vui lòng chọn ít nhất 2 ảnh!', 'error');
        return;
    }

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('images', file);
    });

    loading.classList.add('show');
    hideMessage();
    resultSection.classList.remove('show');
    stitchBtn.disabled = true;

    try {
        const response = await fetch('/stitch', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok && data.success) {
            showMessage(data.message, 'success');
            if (data.image_data) {
                // Hiển thị ảnh trả về trực tiếp (base64 data URL)
                resultImage.src = data.image_data;
                downloadBtn.href = data.image_data;
            } else {
                // Fallback dùng endpoint kết quả theo id
                resultImage.src = `/result/${data.result_id}`;
                downloadBtn.href = `/result/${data.result_id}`;
            }
            resultSection.classList.add('show');
        } else {
            showMessage(data.error || 'Có lỗi xảy ra!', 'error');
        }
    } catch (error) {
        showMessage('Không thể kết nối đến server!', 'error');
    } finally {
        loading.classList.remove('show');
        stitchBtn.disabled = false;
    }
});

function showMessage(text, type) {
    message.textContent = text;
    message.className = `message ${type} show`;
}

function hideMessage() {
    message.classList.remove('show');
}

// Expose removeImage to global scope
window.removeImage = removeImage;


