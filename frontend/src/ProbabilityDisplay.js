import { useSelector } from "react-redux";
import { Bar } from "react-chartjs-2";
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale } from 'chart.js';

ChartJS.register(BarElement, CategoryScale, LinearScale);

export default function ProbabilityDisplay() {
  // Default to an empty array if probabilities are undefined
  const probabilities = useSelector((state) => state.probabilities) || [];

  console.log("YOOOOOOOOo " + probabilities)
  // Prepare data for the histogram
  const data = {
    labels: ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    datasets: [
      {
        label: "Probability (%)",
        data: probabilities.length > 0 
          ? probabilities.map(prob => (Math.exp(prob) / probabilities.reduce((a, b) => a + Math.exp(b), 0)) * 100)
          : Array(10).fill(0), // Display 0% for all digits if probabilities is empty
        backgroundColor: "rgba(75, 192, 192, 0.6)",
        borderColor: "rgba(75, 192, 192, 1)",
        borderWidth: 1,
      },
    ],
  };

  const options = {
    scales: {
      y: {
        beginAtZero: true,
        max: 100,
        title: {
          display: true,
          text: "Probability (%)",
        },
      },
    },
  };

  return (
    <div className="probability-display">
      <h2>Prediction Probabilities</h2>
      {probabilities.length > 0 ? (
        <Bar data={data} options={options} />
      ) : (
        <p>Draw something on the canvas to see the prediction.</p>
      )}
    </div>
  );
}
