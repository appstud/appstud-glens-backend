import { WebSocketGateway } from '../../../WebSocketGateway'
import { WebSocketClient } from '../../../entities/WebSocketClient'
import { ProcessingType } from '../../../../../domain/models/processing/nested/ProcessingType'
import { ImageProcessingService } from '../../../../../domain/services/ImageProcessingService'
import { WebSocketMessage } from '../../../entities/WebSocketMessage'
import { ProcessingFlux } from '../../../../../domain/models/processing/ProcessingFlux'
import { IPeopleStorage } from '../../../../../data/IPeopleStorage'
import {DataService} from "../../../../../domain/services/DataService";
import {FaceDataProcessingService} from "../../../../../domain/services/FaceDataProcessingService";

export class FaceDataController {
    private service: FaceDataProcessingService

    constructor(
        gateway: WebSocketGateway,
        service: FaceDataProcessingService
    ) {
        this.service = service
        gateway.on(
            'message:v1:face:save',
            this.storeData.bind(this)
        )
    }

    initialize(
        client: WebSocketClient,
        callback = (message: string) => client.raw(message)
    ): ProcessingFlux {
        const flux = this.service.create('')
        this.service.register(flux, callback)
        client.add(flux)
        return flux
    }

    storeData(client: WebSocketClient, message: WebSocketMessage) {
        const flux =
            client.flux(ProcessingType.FACE_DATA_PROCESSING) ||
            this.initialize(client, (message: string) => client.raw(message))
        console.log(message)
        this.service.storeFaceData(message)
    }
}
