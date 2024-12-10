$(document).ready(function () {
    $("#upload").click(function (event) {
        console.log('1111');



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
                $('#file').val('')
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