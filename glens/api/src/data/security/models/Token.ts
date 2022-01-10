import { DomainAccessRole } from '../../../domain/models/processing/nested/DomainAccessRole'

export class Token {
    role: DomainAccessRole = DomainAccessRole.UNKNOWN
    refresh = false
}
