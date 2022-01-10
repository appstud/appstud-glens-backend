import { DomainAccessRole } from '../processing/nested/DomainAccessRole'

export default class Authentication {
    authority: DomainAccessRole
    issuedAt: Date
    expiresAt: Date
    subject: string
    isRefreshToken: boolean = false

    constructor(
        authority: DomainAccessRole,
        issuedAt: Date,
        expiresAt: Date,
        subject: string,
        isRefreshToken: boolean
    ) {
        this.authority = authority
        this.issuedAt = issuedAt
        this.expiresAt = expiresAt
        this.subject = subject
        this.isRefreshToken = isRefreshToken
    }
}
