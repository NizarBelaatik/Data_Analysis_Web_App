$(document).ready(function () {
    $("#upload").click(function (event) {

        var formData = new FormData();
        var csrfToken = $("[name=csrfmiddlewaretoken]").val();
        var file = $("#file")[0].files[0];
        formData.append('csrfmiddlewaretoken', csrfToken);
        formData.append('file', file);
        $.ajax({
            url: "/upload_file/",
            type: "POST",
            data:formData,
            processData: false, 
            contentType: false,
            success: function(data){
                console.log('success');
                console.log('status',data.status);
                $('#file').val('');
                window.location.reload();
            },
            
        });

    });
  
});


$(document).ready(function () {
    $("[name=show_type]").click(function (event) {
        
        var formData = new FormData();
        var csrfToken = $("[name=csrfmiddlewaretoken]").val();
        var show_type = $(this).val();
        formData.append('csrfmiddlewaretoken', csrfToken);
        formData.append('show_type', show_type);
        $.ajax({
            url: "/show_table_type/",
            type: "POST",
            data:formData,
            processData: false, 
            contentType: false,
            success: function(data){
                console.log('success');
                console.log('status',data.status);
                $('#table').html(data.html)
            },
            
        });

    });
  
});


$(document).ready(function () {

    function loadChart(url, chartContainerId, errorMessageId) {
        $.ajax({
            url: url,
            method: 'GET',
            success: function(response) {
                
                var img = $('<img>').attr('src', 'data:image/png;base64,' + response.chart);
                $('#' + chartContainerId).html(img);
                $('#' + errorMessageId).text('');
                if(response.summary_stats){

                
                    const summaryStats = response.summary_stats;
                    let statsHtml ='';

                    // Loop through each feature in the summary statistics
                    for (let key in summaryStats) {
                        // Start a new row for each feature
                        statsHtml += `<tr><td><strong>${key}</strong></td>`;

                        // Loop through the statistics for the feature (count, mean, etc.)
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

                    // Insert the generated rows into the table body
                    var htmlfinal=`<h3>Summary Statistics:</h3>
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
                            </table>`
                    $('#summary-stats ').html(htmlfinal);
                }
                
            },
            error: function(xhr, status, error) {
                $('#' + errorMessageId).text('An error occurred: ' + error);

            }
        });
    }

    
    
});

