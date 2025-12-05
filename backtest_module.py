
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def run_backtest(
    df_backtest: pd.DataFrame,
    initial_capital: float = 100000,
    z_entry: float = 2.0,
    z_exit: float = 0.5,
    stop_loss_pct: float = 0.10,
    unit_size: int = 1000   # số đơn vị spread size = 1000
):
    """
    Backtest chiến lược Pair Trading có kết hợp tín hiệu LSTM Trend.

    df_backtest yêu cầu có các cột:
        - 'Z_Score'
        - 'Spread'
        - 'LSTM_Trend'
    """

    # --- 1. Khởi tạo ---
    cash = initial_capital
    position = 0        # 0=No position, 1=Long, -1=Short
    entry_price = 0
    equity_curve = []
    trade_log = []

    print(f"Bắt đầu Backtest trên {len(df_backtest)} ngày...")

    # --- 2. Vòng lặp mô phỏng ---
    for i in range(len(df_backtest)):
        date = df_backtest.index[i]
        row = df_backtest.iloc[i]

        z = row['Z_Score']
        spread = row['Spread']
        signal = row['LSTM_Trend']

        action = "HOLD"

        # =======================
        # 1) Không có vị thế
        # =======================
        if position == 0:

            if z < -z_entry and signal == 1:  # mở LONG
                position = 1
                entry_price = spread
                action = "OPEN LONG"
                trade_log.append(f"{date}: LONG Spread tại {spread:.4f} (Z={z:.2f})")

            elif z > z_entry and signal == -1:  # mở SHORT
                position = -1
                entry_price = spread
                action = "OPEN SHORT"
                trade_log.append(f"{date}: SHORT Spread tại {spread:.4f} (Z={z:.2f})")

        # =======================
        # 2) Đang LONG
        # =======================
        elif position == 1:
            pnl_temp = (spread - entry_price) * unit_size

            # take profit
            if z > -z_exit:
                cash += pnl_temp
                trade_log.append(f"{date}: TP LONG. PnL: {pnl_temp:.2f}")
                position = 0
                action = "CLOSE LONG TP"

            # stop loss
            elif pnl_temp < -abs(entry_price * stop_loss_pct * unit_size):
                cash += pnl_temp
                trade_log.append(f"{date}: SL LONG. PnL: {pnl_temp:.2f}")
                position = 0
                action = "CLOSE LONG SL"

        # =======================
        # 3) Đang SHORT
        # =======================
        elif position == -1:
            pnl_temp = (entry_price - spread) * unit_size

            # take profit
            if z < z_exit:
                cash += pnl_temp
                trade_log.append(f"{date}: TP SHORT. PnL: {pnl_temp:.2f}")
                position = 0
                action = "CLOSE SHORT TP"

            # stop loss
            elif pnl_temp < -abs(entry_price * stop_loss_pct * unit_size):
                cash += pnl_temp
                trade_log.append(f"{date}: SL SHORT. PnL: {pnl_temp:.2f}")
                position = 0
                action = "CLOSE SHORT SL"

        # =======================
        # 4) Mark-to-Market
        # =======================
        unrealized = 0
        if position == 1:
            unrealized = (spread - entry_price) * unit_size
        elif position == -1:
            unrealized = (entry_price - spread) * unit_size

        equity_curve.append(cash + unrealized)

    # =======================
    # 5) Kết quả trả về
    # =======================
    df_out = df_backtest.copy()
    df_out["Equity"] = equity_curve

    # Reset index nếu cần
    if 'Date' not in df_out.columns:
        df_out = df_out.reset_index()
        # Đổi tên cột index thành 'Date' nếu cần
        if 'index' in df_out.columns:
            df_out.rename(columns={'index': 'Date'}, inplace=True)

    # Đảm bảo cột Date đúng định dạng ngày tháng (để vẽ biểu đồ không bị lỗi)
    df_out['Date'] = pd.to_datetime(df_out['Date'])

    final_return = (equity_curve[-1] - initial_capital) / initial_capital * 100

    print("-" * 40)
    print(f"Vốn ban đầu: ${initial_capital:,.2f}")
    print(f"Vốn cuối kỳ: ${equity_curve[-1]:,.2f}")
    print(f"Lợi nhuận tổng: {final_return:.2f}%")
    print(f"Tổng số giao dịch: {len(trade_log)}")
    print("-" * 40)

    return df_out, trade_log, final_return




def plot_spread_signals(df_bt, z_entry=2.0, z_exit=0.1):
    """
    Vẽ biểu đồ Spread và các điểm entry/exit dựa trên Z-Score.

    Args:
        df_bt (DataFrame): dữ liệu backtest có cột Date, Spread, Z_Score
        z_entry (float): ngưỡng vào lệnh (vd: >2: short, <−2: long)
        z_exit (float): ngưỡng thoát lệnh (vd: |Z| < 0.1)
    """

    plt.figure(figsize=(16, 6))

    # Spread line
    plt.plot(df_bt['Date'], df_bt['Spread'], 
             label='Spread', color='black', alpha=0.6, linewidth=1)

    # Entry & Exit
    entry_long  = df_bt[df_bt['Z_Score'] < -z_entry]
    entry_short = df_bt[df_bt['Z_Score'] >  z_entry]
    exit_points = df_bt[df_bt['Z_Score'].abs() < z_exit]

    # Vẽ tín hiệu
    plt.scatter(entry_long['Date'], entry_long['Spread'],
                marker='^', color='green', s=100,
                label='LONG Entry (Mua)', zorder=5)

    plt.scatter(entry_short['Date'], entry_short['Spread'],
                marker='v', color='red', s=100,
                label='SHORT Entry (Bán)', zorder=5)

    plt.scatter(exit_points['Date'], exit_points['Spread'],
                marker='o', color='blue', s=40, alpha=0.6,
                label='Exit (Chốt lời)', zorder=4)

    # Title, legend, aesthetics
    plt.title('Biến động Spread và Điểm vào/thoát lệnh - Pair Trading Strategy',
              fontsize=14, fontweight='bold')

    plt.ylabel('Giá trị Spread')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show()





def plot_zscore_signals(df_bt, z_entry=2.0):
    """
    Vẽ biểu đồ Z-Score, ngưỡng entry/exit và highlight vùng cực trị.

    Args:
        df_bt (DataFrame): dữ liệu có cột Date và Z_Score
        z_entry (float): ngưỡng Z-score để vào lệnh (vd: ±2)
    """

    plt.figure(figsize=(16, 9))

    # Plot Z-Score
    plt.plot(df_bt['Date'], df_bt['Z_Score'],
             label='Z-Score', color='steelblue', linewidth=1.5)

    # Threshold lines
    plt.axhline(z_entry, color='red', linestyle='--',
                linewidth=1.5, label=f'Ngưỡng Bán (+{z_entry})')
    plt.axhline(-z_entry, color='green', linestyle='--',
                linewidth=1.5, label=f'Ngưỡng Mua (-{z_entry})')
    plt.axhline(0, color='black', linestyle='-', linewidth=1,
                label='Trung bình (Mean)')

    # Highlight extreme zones
    plt.fill_between(df_bt['Date'], z_entry, df_bt['Z_Score'],
                     where=(df_bt['Z_Score'] >= z_entry),
                     facecolor='red', alpha=0.3)

    plt.fill_between(df_bt['Date'], -z_entry, df_bt['Z_Score'],
                     where=(df_bt['Z_Score'] <= -z_entry),
                     facecolor='green', alpha=0.3)

    # Labels & layout
    plt.title('Tín hiệu Z-Score (Mean Reversion)',
              fontsize=14, fontweight='bold')
    plt.ylabel('Độ lệch chuẩn (Sigma)')
    plt.xlabel('Thời gian')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()
