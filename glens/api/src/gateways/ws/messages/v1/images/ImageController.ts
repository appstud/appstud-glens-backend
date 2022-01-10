import { WebSocketGateway } from '../../../WebSocketGateway'
import { WebSocketClient } from '../../../entities/WebSocketClient'
import { ProcessingType } from '../../../../../domain/models/processing/nested/ProcessingType'
import { ImageProcessingService } from '../../../../../domain/services/ImageProcessingService'
import { WebSocketMessage } from '../../../entities/WebSocketMessage'
import { ProcessingFlux } from '../../../../../domain/models/processing/ProcessingFlux'
import { IPeopleStorage } from '../../../../../data/IPeopleStorage'

export class ImageController {
    private service: ImageProcessingService
    private storage: IPeopleStorage[]

    constructor(
        gateway: WebSocketGateway,
        service: ImageProcessingService,
        storage: IPeopleStorage[]
    ) {
        this.service = service
        this.storage = storage
        // data is returned back to the client
        gateway.on('message:v1:image:process', this.processImage.bind(this))
        // data is returned back to the client and saved on the server (Excel file for now)
        gateway.on(
            'message:v1:image:process:save',
            this.processImageAndSave.bind(this)
        )
    }

    initialize(
        client: WebSocketClient,
        callback = (message: string) => client.raw(message)
    ): ProcessingFlux {
        // TODO: Customer & authentication
        const flux = this.service.create('')
        this.service.register(flux, callback)
        client.add(flux)
        return flux
    }

    processImage(client: WebSocketClient, message: WebSocketMessage) {
        const flux =
            client.flux(ProcessingType.IMAGE_PROCESSING) ||
            this.initialize(client, (message: string) => client.raw(message))
        this.service.process(flux, message)
    }

    processImageAndSave(client: WebSocketClient, message: WebSocketMessage) {
        const flux =
            client.flux(ProcessingType.IMAGE_PROCESSING) ||
            this.initialize(client, (message: string) => {
                client.raw(message)
                for (let st of this.storage) st.savePeopleData(message)
            })
        this.service.process(flux, message)
    }

}
