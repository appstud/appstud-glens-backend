import * as IORedis from 'ioredis'
import { Redis } from './Redis'
import { Environment } from '../../environment'
import { IMessageBusProvider } from '../../domain/providers/IMessageBusProvider'

export class MessageBus implements IMessageBusProvider {
    private publisher: IORedis.Redis
    private subscriber: IORedis.Redis
    private channels: {
        [k: string]: (channel: string, message: string) => void
    } = {}

    constructor(environment: Environment) {
        this.publisher = new Redis(environment)
        this.subscriber = new Redis(environment)
        this.subscriber.on('message', this.received.bind(this))
    }

    received(channel: string, message: string) {
        const handler = this.channels[channel]
        if (handler) handler(channel, message)
    }

    subscribe(
        channel: string,
        handler: (channel: string, message: string) => void
    ): () => void {
        this.channels[channel] = handler
        this.subscriber.subscribe(channel)
        return this.unsubscribe.bind(this, channel)
    }

    unsubscribe(channel: string) {
        delete this.channels[channel]
        this.subscriber.unsubscribe(channel)
    }

    publish(channel: string, message: {}) {
        this.publisher.publish(channel, JSON.stringify(message))
    }
}
