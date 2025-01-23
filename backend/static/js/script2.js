function loadChart(url, chartContainerId, errorMessageId) {
    // Show the loader before starting the request
    $('#loader').show();
    
    $.ajax({
        url: url,
        method: 'GET',
        success: function(response) {
            // Hide the loader after the data is successfully loaded
            $('#loader').hide();
            
            var img = $('<img>').attr('src', 'data:image/png;base64,' + response.chart);
            $('#' + chartContainerId).html(img);
            $('#' + errorMessageId).text('');
            
            if(response.summary_stats){
                const summaryStats = response.summary_stats;
                let statsHtml = '';

                // Loop through each feature in the summary statistics
                for (let key in summaryStats) {
                    statsHtml += `<tr><td><strong>${key}</strong></td>`;

                    // Loop through the statistics for the feature
                    let featureStats = summaryStats[key];
                    statsHtml += `<td>${featureStats['count']}</td>
                                   <td>${featureStats['mean']}</td>
                                   <td>${featureStats['min']}</td>
                                   <td>${featureStats['25%']}</td>
                                   <td>${featureStats['50%']}</td>
                                   <td>${featureStats['75%']}</td>
                                   <td>${featureStats['max']}</td>
                                   <td>${featureStats['std']}</td></tr>`;
                }

                var htmlfinal = `<h3>Summary Statistics:</h3>
                                 <table id="stats-table" border="1">
                                     <thead>
                                         <tr>
                                             <th>Feature</th>
                                             <th>Count</th>
                                             <th>Mean</th>
                                             <th>Min</th>
                                             <th>25%</th>
                                             <th>50%</th>
                                             <th>75%</th>
                                             <th>Max</th>
                                             <th>Std</th>
                                         </tr>
                                     </thead>
                                     <tbody>${statsHtml}</tbody>
                                 </table>`;
                $('#summary-stats').html(htmlfinal);
            }
        },
        error: function(xhr, status, error) {
            // Hide the loader if there is an error
            $('#loader').hide();
            $('#' + errorMessageId).text('An error occurred: ' + error);
        }
    });
}



function fetchDataAndRenderChart() {
    // Show the loader before starting the request
    $('#loader').show();
    $('#show_data_id').hide();
    const store_id = $('#store_id').val();
    const date_column = 'Date';
    const target_column = 'Sales';

    $('#chart-container' ).html(`
        <div id="chart-container" style="width: 80%; height: 400px; margin-top: 20px;">
            <canvas id="myChart"></canvas>
        </div>

        <div id="forecast-chart-container">
        </div>`);

    $.ajax({
        url: "/TSF_ARIMA/"+FileId,
        type: 'GET',
        data: {
            store_id: store_id,
            date_column: date_column,
            target_column: target_column
        },
        success: function(response) {
            // Hide the loader after the data is loaded
            $('#loader').hide();
            
            const forecast = response.forecast;
            const dates = response.dates;

            $('#forecast-list').empty();
            forecast.forEach((value, index) => {
                $('#forecast-list').append(`<h3>12-Week Forecast:</h3>
                    <ul id="forecast-list"></ul>
                    <li>Week ${index + 1}: ${value}</li>`);
            });

            const ctx = document.getElementById('myChart').getContext('2d');
            const chartData = {
                labels: dates,
                datasets: [{
                    label: 'Forecasted Sales',
                    data: forecast,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    fill: false
                }]
            };

            const chartOptions = {
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            };

            const myChart = new Chart(ctx, {
                type: 'line',
                data: chartData,
                options: chartOptions
            });

            $('#forecast-chart-container').append(`<img src="data:image/png;base64,${response.image}" alt="Sales Forecast Chart" />`);
        },
        error: function(error) {
            // Hide the loader on error
            $('#loader').hide();
            console.error('Error fetching data:', error);
        }
    });
}
$('#load-sales-trend').click(function() {
    $('#show_data_id').hide();
    var cContainteId = ""
    $('#chart-container' ).html(`
                    <h2>Sales Trend</h2>
            <div id="sales-trend-chart"></div>
            <div id="summary-stats"></div>
        `);
    loadChart("/eda_view/" + FileId, "sales-trend-chart", "error-message");
});

$('#load-correlation-heatmap').click(function() {
    $('#show_data_id').hide();
    $('#chart-container' ).html(`<h2>Correlation Heatmap</h2>
            <div id="correlation-heatmap"></div>`);
    loadChart("/correlation_view/" + FileId, "correlation-heatmap", "error-message");
});

$('#load-prediction-chart').click(function() {
    $('#show_data_id').hide();
    $('#chart-container' ).html(`<h2>Sales Prediction Model</h2>
            <div id="model-results-chart"></div>
            <div id="model-results">
                <p id="rmse"></p>
                <p id="mae"></p>
            </div>`);
    $('#loader').show();
    $.ajax({
        url: "/training_view/" + FileId,
        method: 'GET',
        success: function(response) {
            $('#rmse').text(`RMSE: ${response.rmse}`);
            $('#mae').text(`MAE: ${response.mae}`);

            var img = $('<img>').attr('src', 'data:image/png;base64,' + response.chart);
            $('#model-results-chart').html(img);
            $('#loader').hide();
        },
        error: function() {
            $('#model-results').html('<p>Error training the model.</p>');
            $('#loader').hide();
        }
    });
});



$('#show_data_button').click(function() {
    $('#chart-container' ).html(``);
    $('#show_data_id').show();
    loadChart("/correlation_view/" + FileId, "correlation-heatmap", "error-message");
});






function searchTable() {
    // Get the search input value
    let input = document.getElementById('search-input');
    let filter = input.value.toLowerCase();

    // Get the selected column index from the dropdown
    let columnSelect = document.getElementById('column-select');
    let selectedColumnIndex = columnSelect.value;

    // Get the table and all its rows
    let table = document.getElementById('stats-table');
    let rows = table.getElementsByTagName('tr'); // Get all table rows

    // Loop through each table row (skip the header row)
    for (let i = 1; i < rows.length; i++) { // Start from 1 to skip the header row
        let row = rows[i];
        let cells = row.getElementsByTagName('td'); // Get all table cells in this row

        let rowMatches = false;

        // If 'All' is selected, search across all columns
        if (selectedColumnIndex === 'all') {
            // Loop through each cell in the row
            for (let j = 0; j < cells.length; j++) {
                let cellText = cells[j].textContent.toLowerCase();
                if (cellText.indexOf(filter) > -1) {
                    rowMatches = true;
                    break; // No need to check further cells
                }
            }
        } else {
            // If a specific column is selected, check that one column only
            let cellText = cells[selectedColumnIndex] ? cells[selectedColumnIndex].textContent.toLowerCase() : '';
            if (cellText.indexOf(filter) > -1) {
                rowMatches = true;
            }
        }

        // Show or hide the row based on whether it matches
        if (rowMatches) {
            row.style.display = ''; // Show row
        } else {
            row.style.display = 'none'; // Hide row
        }
    }
}