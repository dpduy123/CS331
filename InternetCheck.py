import InternetCheck
# pip3 install speedtest-cli

def test_speed():
    st = InternetCheck.Speedtest()
    st.get_best_server()
    download_speed = st.download() / 1_000_000  
    upload_speed = st.upload() / 1_000_000  
    ping_result = st.results.ping

    print(f"Download Speed: {download_speed:.2f} Mbps")
    print(f"Upload Speed: {upload_speed:.2f} Mbps")
    print(f"Ping: {ping_result:.2f} ms")

if __name__ == "__main__":
    test_speed()    