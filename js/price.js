// JavaScript function to toggle menu display
function toggleMenu() {
  const menuContent = document.getElementById("mySidenav");
  menuContent.style.width =
    menuContent.style.width === "250px" ? "0px" : "250px";
}
document.querySelectorAll(".nav-link").forEach((link) => {
  link.addEventListener("click", function () {
    document
      .querySelectorAll(".nav-link")
      .forEach((link) => link.classList.remove("active"));
    this.classList.add("active");
  });
});

// showing pop-up window
function showDiv() {
  document.getElementById("pop-up").style.display = "block";
}

// showing pop-up window
function hideDiv() {
  document.getElementById("pop-up").style.display = "None";
}

// Generate random data function within the range 30-80
function getRandomData() {
  let data = [];
  for (let i = 0; i < 12; i++) {
    // For 12 months
    data.push(Math.floor(Math.random() * 51) + 30); // Random value between 30 and 80
  }
  return data;
}

// Define datasets for different years
const yearData = {
  2024: {
    special: getRandomData(),
    wellMilled: getRandomData(),
    premium: getRandomData(),
    regular: getRandomData(),
  },
  2023: {
    special: getRandomData(),
    wellMilled: getRandomData(),
    premium: getRandomData(),
    regular: getRandomData(),
  },
  2022: {
    special: getRandomData(),
    wellMilled: getRandomData(),
    premium: getRandomData(),
    regular: getRandomData(),
  },
  2021: {
    special: getRandomData(),
    wellMilled: getRandomData(),
    premium: getRandomData(),
    regular: getRandomData(),
  },
};

// Evaluate datapoints for a given month and return an array of rankings
function getRankingsForMonth(special, wellMilled, premium, regular) {
  let dataPoints = [
    { label: "Special", value: special },
    { label: "Well Milled", value: wellMilled },
    { label: "Premium", value: premium },
    { label: "Regular", value: regular },
  ];

  // Sort data points based on value (descending order)
  dataPoints.sort((a, b) => b.value - a.value);

  // Assign rankings based on sorted data
  dataPoints[0].evaluation = "GREAT";
  dataPoints[1].evaluation = "GOOD";
  dataPoints[2].evaluation = "AVERAGE";
  dataPoints[3].evaluation = "NOT GOOD";

  return dataPoints;
}

// Chart configuration
const ctx = document.getElementById("retail-price-chart");

let chart = new Chart(ctx, {
  type: "line",
  data: {
    labels: [
      "Jan",
      "Feb",
      "Mar",
      "Apr",
      "May",
      "Jun",
      "Jul",
      "Aug",
      "Sep",
      "Oct",
      "Nov",
      "Dec",
    ],
    datasets: [
      {
        label: "Special",
        data: yearData[2024].special, // Initial data for "Special" (2024 as default)
        borderWidth: 1,
        pointBackgroundColor: "#dcf16f",
        borderColor: "#dcf16f",
        radius: 2,
      },
      {
        label: "Well Milled",
        data: yearData[2024].wellMilled, // Initial data for "Well Milled" (2024 as default)
        borderWidth: 1,
        pointBackgroundColor: "rgba(153, 102, 255, 1)",
        borderColor: "rgba(153, 102, 255, 1)",
        radius: 2,
      },
      {
        label: "Premium",
        data: yearData[2024].premium, // Initial data for "Premium" (2024 as default)
        borderWidth: 1,
        pointBackgroundColor: "rgba(255, 99, 132, 1)",
        borderColor: "rgba(255, 99, 132, 1)",
        radius: 2,
      },
      {
        label: "Regular",
        data: yearData[2024].regular, // Initial data for "Regular" (2024 as default)
        borderWidth: 1,
        pointBackgroundColor: "rgba(54, 162, 235, 1)",
        borderColor: "rgba(54, 162, 235, 1)",
        radius: 2,
      },
    ],
  },
  options: {
    plugins: {
      legend: {
        display: false, // Hides the legend
      },
      tooltip: {
        enabled: true, // Show tooltips on hover
        callbacks: {
          // Custom tooltip to show evaluation
          label: function (tooltipItem) {
            const monthIndex = tooltipItem.dataIndex;

            // Get the data points for the current month
            const rankings = getRankingsForMonth(
              chart.data.datasets[0].data[monthIndex],
              chart.data.datasets[1].data[monthIndex],
              chart.data.datasets[2].data[monthIndex],
              chart.data.datasets[3].data[monthIndex]
            );

            // Find the current dataset and corresponding ranking
            const datasetLabel = tooltipItem.dataset.label;
            const evaluation = rankings.find(
              (item) => item.label === datasetLabel
            ).evaluation;

            // Return the formatted tooltip content
            return `${datasetLabel}: ${tooltipItem.raw} (${evaluation})`;
          },
        },
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        ticks: {
          display: false,
        },
      },
    },
  },
});

// Add event listener to dropdown menu to update chart data
document.getElementById("year-options").addEventListener("change", function () {
  const selectedYear = this.value; // Get the selected year

  // Update the chart data based on the selected year
  chart.data.datasets[0].data = yearData[selectedYear].special;
  chart.data.datasets[1].data = yearData[selectedYear].wellMilled;
  chart.data.datasets[2].data = yearData[selectedYear].premium;
  chart.data.datasets[3].data = yearData[selectedYear].regular;

  // Re-render the chart
  chart.update();
});
