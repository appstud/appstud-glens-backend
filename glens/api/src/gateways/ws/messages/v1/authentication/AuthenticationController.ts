import { WebSocketGateway } from '../../../WebSocketGateway'
import { WebSocketClient } from '../../../entities/WebSocketClient'
import { WebSocketMessage } from '../../../entities/WebSocketMessage'

export class AuthenticationController {
    constructor(gateway: WebSocketGateway) {
        gateway.on(
            'message:v1:authentication:authenticate',
            this.processImage.bind(this)
        )
    }

    processImage(client: WebSocketClient, message: WebSocketMessage) {}
}
