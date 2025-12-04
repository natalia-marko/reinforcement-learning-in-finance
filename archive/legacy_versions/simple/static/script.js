let chartInstance = null;

async function analyzePortfolio() {
    const input = document.getElementById('tickerInput');
    const btn = document.getElementById('analyzeBtn');
    const btnText = document.getElementById('btnText');
    const loader = document.getElementById('loader');
    const resultsSection = document.getElementById('resultsSection');

    const tickers = input.value.split(',').map(t => t.trim()).filter(t => t.length > 0);

    if (tickers.length < 2) {
        alert("Please enter at least 2 tickers.");
        return;
    }

    // UI Loading State
    btn.disabled = true;
    btnText.classList.add('hidden');
    loader.classList.remove('hidden');
    resultsSection.classList.add('hidden');

    try {
        const response = await fetch('http://127.0.0.1:8000/analyze', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ tickers: tickers }),
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Analysis failed');
        }

        const data = await response.json();
        displayResults(data.weights);

    } catch (error) {
        alert(`Error: ${error.message}`);
    } finally {
        // Reset UI
        btn.disabled = false;
        btnText.classList.remove('hidden');
        loader.classList.add('hidden');
    }
}

function displayResults(weights) {
    const resultsSection = document.getElementById('resultsSection');
    const tableBody = document.querySelector('#weightsTable tbody');

    resultsSection.classList.remove('hidden');
    tableBody.innerHTML = '';

    // Prepare data for chart
    const labels = Object.keys(weights);
    const values = Object.values(weights).map(w => (w * 100).toFixed(2));
    const colors = generateColors(labels.length);

    // Populate Table
    let totalAlloc = 10000; // Example $10k portfolio

    labels.forEach((ticker, index) => {
        const weight = values[index];
        const allocation = (weights[ticker] * totalAlloc).toLocaleString('en-US', { style: 'currency', currency: 'USD' });

        const row = `
            <tr>
                <td><span style="color: ${colors[index]}">●</span> ${ticker}</td>
                <td>${weight}%</td>
                <td>${allocation}</td>
            </tr>
        `;
        tableBody.innerHTML += row;
    });

    // Render Chart
    renderChart(labels, values, colors);
}

function renderChart(labels, data, colors) {
    const ctx = document.getElementById('allocationChart').getContext('2d');

    if (chartInstance) {
        chartInstance.destroy();
    }

    chartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: colors,
                borderWidth: 0,
                hoverOffset: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    callbacks: {
                        label: function (context) {
                            return `${context.label}: ${context.parsed}%`;
                        }
                    }
                }
            },
            cutout: '70%'
        }
    });
}

function generateColors(count) {
    // HSL colors for premium look
    const colors = [];
    const hueStep = 360 / count;
    for (let i = 0; i < count; i++) {
        colors.push(`hsl(${i * hueStep}, 70%, 60%)`);
    }
    return colors;
}
