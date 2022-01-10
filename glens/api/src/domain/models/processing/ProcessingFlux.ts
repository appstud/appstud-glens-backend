import { v4 as uuid } from 'uuid'
import { ProcessingType } from './nested/ProcessingType'

export class ProcessingFlux {
    id: string = uuid()
    customer: string
    type: ProcessingType
    private message: number = -1

    constructor(customer: string, type: ProcessingType) {
        this.customer = customer
        this.type = type
    }

    get messageID() {
        if (this.message + 1 == Number.MAX_SAFE_INTEGER) this.message = -1
        this.message += 1
        return `${this.id}${this.message}`
    }
}
