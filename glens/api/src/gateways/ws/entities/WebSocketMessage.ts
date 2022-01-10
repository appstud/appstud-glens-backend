export interface WebSocketMessage {
    type: string
    payload: { [k: string]: unknown }
}
