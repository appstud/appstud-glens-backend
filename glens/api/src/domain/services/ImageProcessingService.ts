import { ProcessingFlux } from '../models/processing/ProcessingFlux'
import { ProcessingType } from '../models/processing/nested/ProcessingType'
import { IMessageBusProvider } from '../providers/IMessageBusProvider'
import { WebSocketMessage } from '../../gateways/ws/entities/WebSocketMessage'

export class ImageProcessingService {
    private bus: IMessageBusProvider

    constructor(bus: IMessageBusProvider) {
        this.bus = bus
    }

    create(customer: string): ProcessingFlux {
        return new ProcessingFlux(customer, ProcessingType.IMAGE_PROCESSING)
    }

    register(flux: ProcessingFlux, callback: (message: string) => void) {
        this.bus.subscribe(flux.id, (c, m) => callback(m))
    }

    remove(flux: ProcessingFlux) {
        this.bus.unsubscribe(flux.id)
    }

    process(flux: ProcessingFlux, data: WebSocketMessage) {
        let first_worker_channel = (data.payload.pipeline as string)
            .split('|')[0]
            .trim()
            .split(' ')[0]
        let updated_pipeline = (data.payload.pipeline as string).concat(
            '|',
            flux.id
        )
        this.bus.publish(first_worker_channel, {
            pipeline: updated_pipeline,
            CAM_ID: data.payload.CAM_ID,
            ID_IMG: flux.messageID,
            image: data.payload.image,
            current_time: data.payload.current_time,
        })
    }
}
