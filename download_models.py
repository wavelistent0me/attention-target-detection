import requests

# 要下载的文件及其保存的文件名
files = {
    "https://www.dropbox.com/s/s9y65ajzjz4thve/initial_weights_for_spatial_training.pt": "initial_weights_for_spatial_training.pt",
    "https://www.dropbox.com/s/ye3lyyyzd73afa7/initial_weights_for_temporal_training.pt": "initial_weights_for_temporal_training.pt",
    "https://www.dropbox.com/s/vt8hua06r1yoi2i/model_demo.pt": "model_demo.pt",
    "https://www.dropbox.com/s/nloym5bmvv1v7wr/model_gazefollow.pt": "model_gazefollow.pt",
    "https://www.dropbox.com/s/ywd16kcv06vn93x/model_videoatttarget.pt": "model_videoatttarget.pt",
}

def download_file(url, filename):
    # 将 dropbox 链接转换为直接下载链接
    url = url.replace("www.dropbox.com", "dl.dropboxusercontent.com")
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
        print(f"Downloaded {filename}")
    else:
        print(f"Failed to download {filename} (status code: {response.status_code})")

if __name__ == "__main__":
    for url, name in files.items():
        download_file(url, name)