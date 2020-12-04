var express = require('express')
var csv = require('csv-express')
var app = express()
 
app.get('index.html',function(req,res){
  res.sendFile(path.join(__dirname+'index.html'));
});
app.post('/process_post',function(req,res){
var first_name= req.body.first_name;
var last_name= req.body.last_name; 
var age= req.body.age;
var email= req.body.email;
var password= req.body.password;
var genres= req.body.genres;
var gender= req.body.gender;
res.write('You sent the  first name as "' + req.body.first_name+'".\n');
res.write('You sent the  last name as "' + req.body.last_name+'".\n');
res.write('You sent the age "' + req.body.age+'".\n');
res.write('You sent the  email as "' + req.body.email+'".\n');
res.write('You sent the  password as "' + req.body.password+'".\n');
res.write('Your favorite genre(s) is/are "' + req.body.genres+'".\n');
res.write('You are a "' + req.body.gender+'".\n');

  res.csv([
    ["a", "b", "c"]
  , ["d", "e", "f"]
  ])

 
// Add headers
app.get('/headers', function(req, res) {
  res.csv([
    {"a": first_name, "b":last_name , "c": email},
    {"a": 4, "b": 5, "c": 6}
  ], true)
})
 
// Add response headers and status code (will throw a 500 error code)
app.get('/all-params', function(req, res) {
  res.csv([
     {"a": first_name, "b":last_name , "c": email},
    {"a": 4, "b": 5, "c": 6}
  ], true, {
    "Access-Control-Allow-Origin": "*"
  }, 500)
})
})
 var server = app.listen(8000, function () {  
  var host = server.address().address 
  var port = server.address().port 
  console.log("Example app listening at http://%s:%s", host, port)  
}) 