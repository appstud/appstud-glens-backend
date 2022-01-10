import * as WebSocket from 'ws'
import { Environment } from '../../environment'
import { IModule } from '../../dependencies'
import { forClass } from '../../domain/helpers/Logging'
import { WebSocketClient } from './entities/WebSocketClient'
import { WebSocketMessage } from './entities/WebSocketMessage'
import { IncomingMessage } from 'http'

export class WebSocketGateway implements IModule {
    private logger = forClass('WebSocketGateway')
    private instance: WebSocket.Server
    private environment: Environment

    private handlers: {
        [k: string]: (
            client: WebSocketClient,
            message: WebSocketMessage
        ) => void
    } = {}
    private clients: WebSocketClient[] = []
    private healthCheckInterval?: NodeJS.Timer

    constructor(environment: Environment) {
        this.environment = environment
        this.instance = new WebSocket.Server({
            host: this.host,
            port: this.port,
        })
    }

    private get port() {
        return +this.environment.getConfigurationOrDefault('WS_PORT', 8081)
    }

    private get host() {
        return this.environment.getConfigurationOrDefault('WS_HOST', '0.0.0.0')
    }

    async start() {
        this.instance.on('connection', this.newConnection.bind(this))
        this.healthcheck(true)
        this.logger.info(`Listening on ws://${this.host}:${this.port}`)
    }

    async stop() {
        // TODO(): Gracefully close clients
        this.logger.info('Closing websocket gateway')
        this.healthcheck(false)
        await this.instance.close()
    }

    healthcheck(run: boolean = true) {
        if (this.healthCheckInterval) clearTimeout(this.healthCheckInterval)
        if (run)
            this.healthCheckInterval = setInterval(() => {
                this.clients.forEach((ws: WebSocketClient) => {
                    if (!ws.isAlive) return ws.close()
                    ws.markAsInactive()
                })
            }, 5000)
    }

    on(
        type: string,
        handler: (client: WebSocketClient, message: WebSocketMessage) => void
    ) {
        this.handlers[type] = handler
    }

    newConnection(ws: WebSocket, req: IncomingMessage) {
        const client = new WebSocketClient(ws, req.connection.remoteAddress)
        this.clients.push(client)
        this.logger.info(`${client.identification} Connected client`)
        ws.on('message', (message: string) => this.onMessage(client, message))
        ws.on('close', () => this.close(client))
        ws.on('pong', () => client.markAsActive())
    }

    onMessage(ws: WebSocketClient, message: string) {
        try {
            this.handle(ws, this.toMessage(ws, message))
        } catch (e) {
            // TODO better exception handling
            this.logger.error(`${ws.identification} ${e}`)
            ws.json({ error: e.message || 'Internal server error' })
        }
    }

    toMessage(
        ws: WebSocketClient,
        message: string
    ): WebSocketMessage | undefined {
        if (!message) {
            this.logger.error(`${ws.identification} No message received`)
            return undefined
        }
        const payload = JSON.parse(message)

        if (!payload.type) {
            this.logger.error(
                `${ws.identification} No type received in message`
            )
            return undefined
        }
        return payload
    }

    handle(ws: WebSocketClient, message?: WebSocketMessage) {
        if (!message) return

        const handler = this.handlers[message.type]
        if (!handler)
            return this.logger.error(
                `${ws.identification} No handler for type ${message.type}`
            )
        handler(ws, message)
    }

    close(ws: WebSocketClient) {
        this.logger.info(`${ws.identification} Disconnected`)
    }
}
