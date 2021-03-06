(function() {
    let canvas = document.querySelector("#canvas");
    let context = canvas.getContext("2d");
    canvas.width = 650;
    canvas.height = 300;
    let Mouse = { x: 0, y: 0 };
    let lastMouse = { x: 0, y: 0 };
    context.fillStyle="white";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.color = "black";
    context.lineWidth = 12;
    context.lineJoin = context.lineCap = 'round';

    debug();
    canvas.addEventListener("mousemove", function( e ) {
        lastMouse.x = Mouse.x;
        lastMouse.y = Mouse.y;
        Mouse.x = e.pageX - this.offsetLeft;
        Mouse.y = e.pageY - this.offsetTop;
    }, false );

    canvas.addEventListener("mousedown", function( e ) {
        canvas.addEventListener("mousemove", onPaint, false );
    }, false );

    canvas.addEventListener("mouseup", function() {
        canvas.removeEventListener("mousemove", onPaint, false );
    }, false );
    
    let onPaint = function() {  
        context.lineWidth = context.lineWidth;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.strokeStyle = context.color;

        context.beginPath();
        context.moveTo( lastMouse.x, lastMouse.y );
        context.lineTo( Mouse.x, Mouse.y );
        context.closePath();
        context.stroke();
    };

    function debug() {
        /* CLEAR BUTTON */
        let clearButton = $("#clearButton");

        clearButton.on("click", function() {
            context.clearRect( 0, 0, 664, 373 );
            context.fillStyle="white";
            context.fillRect(0,0,canvas.width,canvas.height);
            $('#result').text(`Hasil Prediksi:`);
        });
    }
}());
