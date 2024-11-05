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

// // Chart configuration
// const ctx = document.getElementById("retail-price-chart");

// let chart = new Chart(ctx, {
//   type: "line",
//   data: {
//     labels: [
//       "Jan",
//       "Feb",
//       "Mar",
//       "Apr",
//       "May",
//       "Jun",
//       "Jul",
//       "Aug",
//       "Sep",
//       "Oct",
//       "Nov",
//       "Dec",
//     ],
//     datasets: [
//       {
//         label: "Special",
//         data: yearData[2024].special, // Initial data for "Special" (2024 as default)
//         borderWidth: 1,
//         pointBackgroundColor: "#dcf16f",
//         borderColor: "#dcf16f",
//         radius: 2,
//       },
//       {
//         label: "Well Milled",
//         data: yearData[2024].wellMilled, // Initial data for "Well Milled" (2024 as default)
//         borderWidth: 1,
//         pointBackgroundColor: "rgba(153, 102, 255, 1)",
//         borderColor: "rgba(153, 102, 255, 1)",
//         radius: 2,
//       },
//       {
//         label: "Premium",
//         data: yearData[2024].premium, // Initial data for "Premium" (2024 as default)
//         borderWidth: 1,
//         pointBackgroundColor: "rgba(255, 99, 132, 1)",
//         borderColor: "rgba(255, 99, 132, 1)",
//         radius: 2,
//       },
//       {
//         label: "Regular",
//         data: yearData[2024].regular, // Initial data for "Regular" (2024 as default)
//         borderWidth: 1,
//         pointBackgroundColor: "rgba(54, 162, 235, 1)",
//         borderColor: "rgba(54, 162, 235, 1)",
//         radius: 2,
//       },
//     ],
//   },
//   options: {
//     plugins: {
//       legend: {
//         display: false, // Hides the legend
//       },
//       tooltip: {
//         enabled: true, // Show tooltips on hover
//         callbacks: {
//           // Custom tooltip to show evaluation
//           label: function (tooltipItem) {
//             const monthIndex = tooltipItem.dataIndex;

//             // Get the data points for the current month
//             const rankings = getRankingsForMonth(
//               chart.data.datasets[0].data[monthIndex],
//               chart.data.datasets[1].data[monthIndex],
//               chart.data.datasets[2].data[monthIndex],
//               chart.data.datasets[3].data[monthIndex]
//             );

//             // Find the current dataset and corresponding ranking
//             const datasetLabel = tooltipItem.dataset.label;
//             const evaluation = rankings.find(
//               (item) => item.label === datasetLabel
//             ).evaluation;

//             // Return the formatted tooltip content
//             return `${datasetLabel}: ${tooltipItem.raw} (${evaluation})`;
//           },
//         },
//       },
//     },
//     scales: {
//       y: {
//         beginAtZero: true,
//         ticks: {
//           display: false,
//         },
//       },
//     },
//   },
// });

// // Add event listener to dropdown menu to update chart data
// document.getElementById("year-options").addEventListener("change", function () {
//   const selectedYear = this.value; // Get the selected year

//   // Update the chart data based on the selected year
//   chart.data.datasets[0].data = yearData[selectedYear].special;
//   chart.data.datasets[1].data = yearData[selectedYear].wellMilled;
//   chart.data.datasets[2].data = yearData[selectedYear].premium;
//   chart.data.datasets[3].data = yearData[selectedYear].regular;

//   // Re-render the chart
//   chart.update();
// });

function submitForm() {
  const month = document.getElementById("month").value;
  const type = document.getElementById("rice_type").value;
  const year = document.getElementById("year").value;

  // Use Fetch API to send data to the server
  fetch("/pricing", {
    method: "POST",
    headers: {
      "Content-Type": "application/x-www-form-urlencoded",
    },
    body: new URLSearchParams({
      month: month,
      type: type,
      year: year,
    }),
  })
    .then((response) => response.json()) // Expecting JSON response
    .then((data) => {
      showModal(data.month, data.type, data.year, data.price); // Show the data in a modal
    })
    .catch((error) => console.error("Error:", error));
}

function showModal(month, type, year, price) {
  const modal = document.getElementById("pop-up");
  document.getElementById("predicted_month").innerText = month;
  document.getElementById("predicted_type").innerText = type;
  document.getElementById("predicted_year").innerText = year;
  document.getElementById("predicted_price").innerText = price;
  modal.style.display = "block";
}

function closeModal() {
  const modal = document.getElementById("pop-up");
  modal.style.display = "none";
}

// Progress bar script
function startTask() {
  // Show the modal when the task starts
  document.getElementById("progress-modal").style.display = "block";

  // Start the task on the server
  fetch("/start-task", { method: "POST" })
    .then((response) => response.json())
    .then((data) => {
      console.log(data.status);
      updateProgress(); // Begin polling for progress updates
    });
}

function updateProgress() {
  fetch("/progress")
    .then((response) => response.json())
    .then((data) => {
      const progressBar = document.getElementById("progress-bar");
      progressBar.style.width = data.progress + "%";
      progressBar.innerText = data.progress + "%";

      // Continue updating if task not completed
      if (data.progress < 100) {
        setTimeout(updateProgress, 500); // Poll every 500 ms
      } else {
        // Close the modal when the task is complete
        setTimeout(() => {
          document.getElementById("progress-modal").style.display = "none";
        }, 1000); // Delay for 1 second before closing
      }
    });
}

const selectElement = document.getElementById("year-options");
const currentYear = new Date().getFullYear();

// Populate year dropdown
for (let i = 0; i <= 5; i++) {
  const option = document.createElement("option");
  option.value = currentYear + i;
  option.textContent = currentYear + i;
  selectElement.appendChild(option);
}

// Fetch the predictions JSON data
async function fetchPredictions() {
  try {
    const response = await fetch(
      "/static/predictions/rice_price_predictions.json"
    );
    if (!response.ok) throw new Error("Failed to load JSON");
    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error fetching predictions:", error);
  }
}

// Populate Year Dropdown
function populateYearDropdown(years) {
  const yearSelect = document.getElementById("year-options");
  yearSelect.innerHTML = ""; // Clear existing options
  years.forEach((year) => {
    const option = document.createElement("option");
    option.value = year;
    option.textContent = year;
    yearSelect.appendChild(option);
  });
}

// Function to get prices for the selected year and type
function getYearlyPrices(data, year) {
  const yearlyPrices = {};
  Object.keys(data).forEach((riceType) => {
    yearlyPrices[riceType] = data[riceType]
      .filter((item) => item.year === parseInt(year))
      .map((item) => ({ month: item.month, price: item.price }));
  });
  return yearlyPrices;
}

// Update Chart with Selected Data
function updateChart(chart, yearlyPrices) {
  const monthlyEvaluations = {}; // Store evaluations for each month

  // Create a temporary storage for prices to evaluate by month
  const monthlyPrices = {};

  // Collect prices for each month
  Object.keys(yearlyPrices).forEach((riceType) => {
    yearlyPrices[riceType].forEach((item) => {
      const month = item.month;
      const price = item.price;

      if (!monthlyPrices[month]) {
        monthlyPrices[month] = [];
      }
      monthlyPrices[month].push({ riceType, price });
    });
  });

  // Evaluate monthly prices and assign evaluations
  Object.keys(monthlyPrices).forEach((month) => {
    const prices = monthlyPrices[month];

    // Sort prices based on the price value
    prices.sort((a, b) => b.price - a.price); // Descending order

    // Store evaluations for the current month
    const evaluations = {};

    // Assign evaluations based on sorted order and missing data
    prices.forEach((entry, index) => {
      const riceType = entry.riceType;
      switch (index) {
        case 0:
          evaluations[riceType] = "VERY GOOD";
          break;
        case 1:
          evaluations[riceType] = "GOOD";
          break;
        case 2:
          evaluations[riceType] = "AVERAGE";
          break;
        case 3:
          evaluations[riceType] = "NOT GOOD";
          break;
      }
    });

    // Handle missing rice types
    const allRiceTypes = Object.keys(yearlyPrices);
    allRiceTypes.forEach((riceType) => {
      if (!evaluations[riceType]) {
        // Determine which evaluation to assign based on the missing rank
        switch (Object.keys(evaluations).length) {
          case 0: // All types are missing
            evaluations[riceType] = "VERY GOOD";
            break;
          case 1: // One type is missing
            evaluations[riceType] = "NOT GOOD"; // Only one type present
            break;
          case 2: // Two types present
            if (!evaluations[allRiceTypes[0]]) {
              evaluations[allRiceTypes[0]] = "VERY GOOD";
            } else if (!evaluations[allRiceTypes[1]]) {
              evaluations[allRiceTypes[1]] = "GOOD";
            } else {
              evaluations[riceType] = "NOT GOOD"; // More than two types present
            }
            break;
          case 3: // Three types present
            evaluations[riceType] = "NOT GOOD"; // Only one type missing
            break;
          default:
            evaluations[riceType] = "NOT GOOD"; // Default fallback
            break;
        }
      }
    });

    // Assign evaluations for the month
    Object.keys(evaluations).forEach((riceType) => {
      monthlyEvaluations[riceType] = evaluations[riceType];
    });
  });

  // Populate chart datasets
  Object.keys(yearlyPrices).forEach((riceType, index) => {
    const prices = yearlyPrices[riceType].map((item) => item.price);
    const labels = yearlyPrices[riceType].map((item) => item.month);
    chart.data.datasets[index].data = prices;
    if (index === 0) chart.data.labels = labels; // Set labels once, assuming months are the same for all types
  });

  // Update chart
  chart.update();

  // Update tooltip callbacks for evaluations
  chart.options.plugins.tooltip.callbacks.label = (tooltipItem) => {
    const riceType = tooltipItem.dataset.label.split(" ")[0].toLowerCase(); // Match key in evaluations
    const price = tooltipItem.raw;
    const evaluation = monthlyEvaluations[riceType] || "NOT EVALUATED"; // Ensure all rice types have evaluations
    return `${
      riceType.charAt(0).toUpperCase() + riceType.slice(1)
    }: ${price} - ${evaluation}`;
  };
}

// Initialize Chart with datasets for each rice type
function initializeChart(riceTypes) {
  const ctx = document.getElementById("retail-price-chart").getContext("2d");

  const colors = {
    regular: "rgba(54, 162, 235, 1)", // Regular
    premium: "rgba(255, 99, 132, 1)", // Premium
    special: "#dcf16f", // Special
    well_milled: "rgba(153, 102, 255, 1)", // Well Milled
  };

  const datasets = riceTypes.map((riceType) => {
    const normalizedRiceType = riceType.replace(/ /g, "_"); // Normalize to match the colors key
    return {
      label: `${
        riceType.charAt(0).toUpperCase() + riceType.slice(1)
      } Price (per kg)`,
      data: [], // Ensure this is filled with actual data later
      borderColor: colors[normalizedRiceType], // Use the defined color for this rice type
      backgroundColor: colors[normalizedRiceType], // Background color can be added if needed
      fill: false, // Set to false to remove area fill
      borderWidth: 2, // Set border width
      pointRadius: 4, // Set point radius
    };
  });

  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [], // Months will be set here
      datasets: datasets,
    },
    options: {
      responsive: true,
      scales: {
        x: {
          title: { display: true, text: "Month" }, // Display the title for the x-axis
          grid: { display: true }, // Show the grid lines for the x-axis
        },
        y: {
          title: { display: false }, // Hide the title for the y-axis
          display: true, // Keep the y-axis displayed
          grid: { display: true }, // Show the grid lines for the y-axis
          ticks: { display: false }, // Hide the tick labels on the y-axis
        },
      },
      plugins: {
        legend: {
          display: false, // Hide the legend
        },
        tooltip: {
          callbacks: {
            label: (tooltipItem) => {
              const riceType = tooltipItem.dataset.label
                .split(" ")[0]
                .toLowerCase(); // Match key in evaluations
              const price = tooltipItem.raw;
              const evaluation =
                monthlyEvaluations[riceType] || "NOT EVALUATED"; // Ensure all rice types have evaluations
              return `${
                riceType.charAt(0).toUpperCase() + riceType.slice(1)
              }: ${price} - ${evaluation}`;
            },
          },
        },
      },
    },
  });
}

// Main function to set up the dropdown and chart
async function setup() {
  const data = await fetchPredictions();
  const riceTypes = Object.keys(data); // ["regular", "premium", "special", "well milled"]
  const years = [...new Set(data[riceTypes[0]].map((item) => item.year))]; // Get unique years

  // Populate the dropdown with available years
  populateYearDropdown(years);

  // Initialize Chart
  const chart = initializeChart(riceTypes);

  // Listen for year selection changes
  document
    .getElementById("year-options")
    .addEventListener("change", (event) => {
      const selectedYear = event.target.value;
      const yearlyPrices = getYearlyPrices(data, selectedYear);

      // Update chart with data for all rice types
      updateChart(chart, yearlyPrices);
    });

  // Trigger initial load for the first available year
  document.getElementById("year-options").dispatchEvent(new Event("change"));
}

// Run setup on load
setup();
