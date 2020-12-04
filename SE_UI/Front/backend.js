var express = require("express");
var app     = express();
var path    = require("path");
var mysql = require('mysql');
var bodyParser = require('body-parser');
app.use(bodyParser.urlencoded({ extended: false }));
app.use(bodyParser.json());
app.set('views','./views');
app.set('view engine','pug');
var con = mysql.createConnection({
  host: "localhost",
  user: "root",
  password: "3443",
  database: "Hospital_Records"
});
app.use(express.static('../Front'))
app.get('/',function(req,res){
  res.sendFile(path.join(__dirname+'/index.html'));
});

app.get('/jerry',function(req,res){
     con.query("SELECT * FROM paitient_details",function(err,rows,fields){
     	if (err) throw err
     	res.render('jerry',{ title:'Patient Details',items: rows})
     })
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
con.connect(function(err){
if (err) throw err;
var sql= "insert into paitient_details(registration_Id,Name,email,sex,Age) values('"+password+"','"+first_name+"','"+email+"','"+gender+"',"+age+")"
con.query(sql, function (err, result) {
    if (err) throw err;
    console.log("1 record inserted");
     res.end();
  });
con.query("select * from paitient_details", function (err, result) {
    if (err) throw err;
    console.log(result);
     res.end();
  });
con.query(" desc paitient_details", function (err, result) {
    if (err) throw err;
    console.log(result);
     res.end();
  });
});
});
var server = app.listen(8000, function () {  
  var host = server.address().address 
  var port = server.address().port 
  console.log("Example app listening at http://%s:%s", host, port)  
}) 
