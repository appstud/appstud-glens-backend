import * as WebSocket from 'ws'

import { noop } from '../../../domain/helpers/Functions'
import { ProcessingFlux } from '../../../domain/models/processing/ProcessingFlux'
import { ProcessingType } from '../../../domain/models/processing/nested/ProcessingType'

//
import { forClass } from '../../../domain/helpers/Logging'
let logger = forClass('WebSocketClient')
//

export class WebSocketClient {
    private _ip: string
    private socket: WebSocket
    // TODO(): Authentication !
    private authenticated: boolean = false
    private alive = true
    private fluxes: { [k: string]: ProcessingFlux } = {}

    constructor(socket: WebSocket, ip: string = 'unknown') {
        this._ip = ip
        this.socket = socket
    }

    get identification() {
        return `[IP: ${this._ip}]`
    }

    get isAlive() {
        return this.alive
    }

    markAsInactive() {
        this.alive = false
        this.socket.ping(noop)
    }

    markAsActive() {
        this.alive = true
    }

    close() {
        this.socket.terminate()
    }

    json(payload: object) {
        this.socket.send(JSON.stringify(payload))
    }

    raw(payload: string) {
        // logger.info(`sending back data ${payload}`)
        this.socket.send(payload)
    }

    flux(type: ProcessingType) {
        return this.fluxes[type]
    }

    add(flux: ProcessingFlux) {
        this.fluxes[flux.type] = flux
        return flux
    }
}
