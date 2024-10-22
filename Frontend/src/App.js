import React, { useState, useEffect } from 'react';
import { Line, Bar, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  BarElement,
  LineElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, BarElement, ArcElement, Title, Tooltip, Legend);

function App() {
  const [priceData, setPriceData] = useState([]);
  const [timeData, setTimeData] = useState([]);
  const [volumeData, setVolumeData] = useState([]);
  const [tradeHistory, setTradeHistory] = useState([]);
  const [trend, setTrend] = useState('상승세');
  const [totalProfit, setTotalProfit] = useState(5000);
  const [previousProfit, setPreviousProfit] = useState(5000);
  const [profitChange, setProfitChange] = useState(0);
  const [isProfitIncreasing, setIsProfitIncreasing] = useState(true);
  const [accuracy, setAccuracy] = useState(85);
  const [profitRate, setProfitRate] = useState(20);

  useEffect(() => {
    const fetchDummyData = () => {
      const dummyPrice = (Math.random() * (60000 - 50000) + 50000).toFixed(2);
      const dummyVolume = (Math.random() * 100).toFixed(2);
      const currentTime = new Date().toLocaleTimeString();
  
      setProfitRate(Math.floor(Math.random() * 100));
      setAccuracy(Math.floor(Math.random() * 100));
  
      const newProfit = totalProfit + Math.floor(Math.random() * 200 - 100);
      setPreviousProfit(totalProfit);
      setTotalProfit(newProfit);
  
      const change = newProfit - previousProfit;
      setProfitChange(change);
      setIsProfitIncreasing(change >= 0);
  
      setPriceData((prevPriceData) => [...prevPriceData, dummyPrice].slice(-20)); // 10개 데이터만 유지
      setTimeData((prevTimeData) => [...prevTimeData, currentTime].slice(-20)); // 10개 시간 데이터만 유지
      setVolumeData((prevVolumeData) => [...prevVolumeData, dummyVolume].slice(-20));
  
      const newTrade = {
        time: currentTime,
        price: dummyPrice,
        volume: dummyVolume,
        action: Math.random() > 0.5 ? 'Buy' : 'Sell',
        result: Math.random() > 0.5 ? 'Win' : 'Lose',
      };
      setTradeHistory((prevHistory) => {
        const updatedHistory = [...prevHistory, newTrade];
        return updatedHistory.length > 15 ? updatedHistory.slice(-15) : updatedHistory;
      });
  
      const recentPrices = [...priceData, dummyPrice].slice(-5);
      const priceDifference = recentPrices[recentPrices.length - 1] - recentPrices[0];
      setTrend(priceDifference > 0 ? '상승세' : '하락세');
    };
    
    // 주기가 10초로 정확히 작동하도록 하기 위해 의존성을 없앰
    const interval = setInterval(fetchDummyData, 700); // 1초(10000ms) 간격으로 데이터 갱신
    return () => clearInterval(interval);
  }, []); // 빈 의존성 배열로 설정해 10초마다만 작동

  const priceChartData = {
    labels: timeData,
    datasets: [
      {
        label: 'Bitcoin Price',
        data: priceData,
        fill: false,
        borderColor: 'blue',
        tension: 0.4,
        borderWidth: 1,
      },
    ],
  };

  const volumeChartData = {
    labels: timeData,
    datasets: [
      {
        label: 'Bitcoin Volume',
        data: volumeData,
        backgroundColor: 'green',
      },
    ],
  };

  const options = {
    scales: {
      x: {
        ticks: {
          maxTicksLimit: 5,
        },
      },
    },
  };

  const profitRateData = {
    labels: ['Profit', 'Loss'],
    datasets: [
      {
        label: 'Profit Rate',
        data: [profitRate, 100 - profitRate],
        backgroundColor: ['#4caf50', '#f44336'],
        hoverOffset: 4,
      },
    ],
  };

  const accuracyData = {
    labels: ['Accuracy', 'Error'],
    datasets: [
      {
        label: 'Prediction Accuracy',
        data: [accuracy, 100 - accuracy],
        backgroundColor: ['#2196f3', '#ff9800'],
        hoverOffset: 4,
      },
    ],
  };

  return (
    <div>
      <div style={styles.header}>
        <h1>Bitcoin Tracker with Dummy Data</h1>
        <div style={styles.totalProfit}>
          <h2 style={styles.fixedProfit}>
            Total Profit: ${totalProfit.toLocaleString()}
            <span style={{ color: isProfitIncreasing ? 'green' : 'red' }}>
              {isProfitIncreasing ? ` (+${profitChange}) ↑` : ` (${profitChange}) ↓`}
            </span>
          </h2>
        </div>
      </div>

      <div style={styles.container}>
        <div style={styles.column}>
          <h2>Bitcoin Price</h2>
          <Line data={priceChartData} options={options} />

          <h2>Bitcoin Volume</h2>
          <Bar data={volumeChartData} options={options} />
        </div>

        <div style={styles.column}>
          <h2>Trade History (Dummy) - Latest 15</h2>
          <table border="1" cellPadding="10" cellSpacing="0">
            <thead>
              <tr>
                <th>Time</th>
                <th>Price (USD)</th>
                <th>Volume</th>
                <th>Action</th> 
                <th>Result</th> 
              </tr>
            </thead>
            <tbody>
              {tradeHistory.map((trade, index) => (
                <tr key={index}>
                  <td>{trade.time}</td>
                  <td>{trade.price}</td>
                  <td>{trade.volume}</td>
                  <td>{trade.action}</td>
                  <td>{trade.result}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div style={styles.column}>
          <h2>Current Trend: {trend}</h2>
          <p>최근 5개의 가격을 기반으로 분석된 결과: {trend}</p>

          <h2>Profit Rate</h2>
          <div style={styles.chartWrapper}>
            <Doughnut data={profitRateData} />
            <p style={styles.chartText}>{profitRate}%</p>
          </div>

          <h2>Prediction Accuracy</h2>
          <div style={styles.chartWrapper}>
            <Doughnut data={accuracyData} />
            <p style={styles.chartText}>{accuracy}%</p>
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingBottom: '20px',
    borderBottom: '2px solid #ccc',
  },
  totalProfit: {
    fontSize: '24px',
    fontWeight: 'bold',
  },
  fixedProfit: {
    display: 'inline-block',
    width: '200px',
    textAlign: 'right',
  },
  container: {
    display: 'flex',
    flexDirection: 'row',
    justifyContent: 'flex-start',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
  },
  column: {
    flex: 1,
    minWidth: '300px',
    marginRight: '10px',
    marginBottom: '20px',
  },
  chartWrapper: {
    position: 'relative',
    width: '300px',
    height: '300px',
    margin: '0 auto',
  },
  chartText: {
    position: 'absolute',
    top: '50%',
    left: '50%',
    transform: 'translate(-50%, -50%)',
    fontSize: '24px',
    fontWeight: 'bold',
  },
};

export default App;
