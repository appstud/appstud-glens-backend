
function MessageBus() {
    //this.logger = new Logger('sockets')
    this.ws
    
    this.connect = () => {

        //this.ws = new WebSocket('ws://localhost:8081')
        this.ws = new WebSocket('ws://172.31.0.2:8081')
        
        //this.logger.log('Connecting ...')
        this.ws.onopen = () => {
            //this.logger.log('Connected')
            //console.log('Connected')
            this.onConnected()
        }    
        this.ws.onmessage = (event) => {
            //this.logger.log('Received message')
            //console.log('Received message')
            this.onMessage(event)
        }
        this.ws.onclose = () => {
            this.ws = null
            //console.log('Connection lost. Reconnecting.')
            //this.logger.log('Connection lost. Reconnecting.')
            this.connect()
        }
    }

    // Noop. Need to be implemented by caller / main
    this.onConnected = () => {}
    this.onMessage = () => {}
    this.send = (message) => {
        if (this.ws)
            this.ws.send(message)
    }
    return this
}



 
// Queue the images to be transmitted,
// servicing the queue by timer, and
// closing the socket and worker when
// the last image has been sent.
let frame, queue = [], done = false, sending=false,timer = setInterval(serviceQueue,43);

message_bus=new MessageBus()
this.message_bus.onMessage = (event) => {
  let result;
  if (event.data){
      //result = JSON.parse(event.data).attributes
      //result = JSON.parse(event.data)
      postMessage(JSON.parse(event.data))
      sending=false
      //console.log(JSON.parse(event.data))
  }   
}
this.message_bus.connect()

function serviceQueue(){
    
    if (sending) {
        queue.pop();
        return;}

    if (queue.length > 0) {
        sending=true;
        
        frame = queue.pop();
        //console.log("sending")
        message_bus.send(frame["data"])
          
       

        
        //console.log('[WORKER]: Sending frame '+ frame.ordinal);
    } else if (done && queue.length === 0) {
        clearInterval(timer);
        socket.close();
        close();
    }
}
// Handle messages from the web page
onmessage = function (e){
   
    queue.push(e);
    /*
    var message = e.data;
    switch (message.type) {
        // Add a frame to the queue
        case 'message:v1:image:process':
            //delete message['type'];
            //console.log('[WORKER]: Received frame '+ message.ordinal);
            queue.push(e);
            break;
        // That's all, folks
        case 'done':
           // console.log('[WORKER]: Done. Closing socket and web worker.');
            done = true;
            break;
    }
    */
};


