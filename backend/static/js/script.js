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