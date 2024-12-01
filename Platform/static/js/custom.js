function myFunction() {
            
        

    $.ajax({
    url: "{% url 'get_json_message' %}" ,
    type: "GET",

    success: function (response) {


     console.log(response);


 
    },
    error: function (error) {
      console.log("Request Failed");
    },
  });


    }
    