


function initCanvas() {
        canvas = document.getElementById('canvas');
        canvas.setAttribute('width', 200);
        canvas.setAttribute('height', 200);
        canvas.setAttribute('id', 'canvas');

        if(typeof G_vmlCanvasManager != 'undefined') {
            canvas = G_vmlCanvasManager.initElement(canvas);
        }
        context = canvas.getContext("2d");
          $('#canvas').mousedown(function(e){
          pressed= true;
          moves.push([e.pageX - this.offsetLeft,
              e.pageY - this.offsetTop,
              false]);
          redraw();
        });

        $('#canvas').mousemove(function(e){
          if(pressed){
              moves.push([e.pageX - this.offsetLeft,
                  e.pageY - this.offsetTop,
                  true]);
            redraw();
          }
        });

        $('#canvas').mouseup(function(e){
          pressed = false;
        });

        $('#canvas').mouseleave(function(e){
          pressed = false;
        });
    }


function redraw(){
  canvas.width = canvas.width; // Limpia el lienzo

  context.strokeStyle = "#000000";
  context.lineJoin = "round";
  context.lineWidth = 15;

  for(var i=0; i < moves.length; i++)
  {
    context.beginPath();
    if(moves[i][2] && i){
      context.moveTo(moves[i-1][0], moves[i-1][1]);
     }else{
      context.moveTo(moves[i][0], moves[i][1]);
     }
     context.lineTo(moves[i][0], moves[i][1]);
     context.closePath();
     context.stroke();
  }
}

function clearArea() {
    context.setTransform(1, 0, 0, 1, 0, 0);
    context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    moves = new Array();
}

function upload() {
    $.getJSON('/get_canvas_data', {
        img:  canvas.toDataURL()
    },function(data) {
  $('#result').text(data.result);
  $('input[name=a]').focus().select();
});
document.getElementById("result").innerHTML = "Predicting, Please wait...";
return false;
}