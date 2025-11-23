from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd

# 1. Danh sách tickers (thay đổi nếu cần)
tickers_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
                'NVDA', 'META', 'JPM', 'V', 'JNJ']

print(f"Bắt đầu tải dữ liệu cho {len(tickers_list)} tickers...")

# 2. Thiết lập ngày: lấy ngày kết thúc là 01/10/2025, start = 10 năm trước
end_date = datetime(2025, 10, 1)
# trừ 10 năm (lưu ý: với ngày như 01/10 an toàn để replace year)
start_date = end_date.replace(year=end_date.year - 10)

# để chắc chắn bao gồm 01/10/2025, thêm 1 ngày vào end (yfinance có thể coi end-exclusive)
end_date_inclusive = end_date + timedelta(days=1)

start_str = start_date.strftime('%Y-%m-%d')      # '2015-10-01'
end_str = end_date_inclusive.strftime('%Y-%m-%d')# '2025-10-02' -> sẽ bao gồm 2025-10-01

print(f"Tải dữ liệu từ {start_str} đến {end_str} (bao gồm {end_date.strftime('%Y-%m-%d')}).")

# 3. Tải dữ liệu với yfinance (dùng start & end)
try:
    # download trả về DataFrame với multiindex columns khi có nhiều ticker
    data = yf.download(tickers_list, start=start_str, end=end_str, progress=True, threads=True)

    if data is None or data.empty:
        print("Không tải được dữ liệu. Vui lòng kiểm tra lại mã tickers hoặc kết nối mạng.")
    else:
        print("Dữ liệu đã được tải thành công (hiển thị 5 dòng đầu):")
        print(data.head())

        # 4. Lưu DataFrame vào file CSV (thay đường dẫn nếu cần)
        file_name = "du_lieu_10_tickers_2015-10-01__2025-10-01.csv"
        data.to_csv(file_name)
        print(f"\nĐã lưu thành công dữ liệu vào file: {file_name}")

except Exception as e:
    print(f"Đã xảy ra lỗi khi tải dữ liệu: {e}")
