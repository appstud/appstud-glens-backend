export interface IMessageBusProvider {
    subscribe(
        channel: string,
        handler: (channel: string, message: string) => void
    ): () => void
    publish(channel: string, message: {}): void
    unsubscribe(channel: string): void
}
