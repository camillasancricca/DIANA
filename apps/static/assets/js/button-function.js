var btnOne = document.getElementById('one');
var btnTwo = document.getElementById('two');
var btnThree = document.getElementById('three');

btnOne.addEventListener('click', function() {
    document.getElementById("test2").style.display = "block";
    document.getElementById("test1").style.display = "none";
});

btnTwo.addEventListener('click', function() {
    document.getElementById("test1").style.display = "block";
    document.getElementById("test2").style.display = "none";
});

btnThree.addEventListener('click', function() {
    document.getElementById("test1").style.display = "block";
    document.getElementById("test2").style.display = "none";
});
