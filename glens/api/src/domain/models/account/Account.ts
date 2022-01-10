import { DomainAccessRole } from '../processing/nested/DomainAccessRole'

export default class Account {
    id: string
    email: string
    password: string // hashed
    createdDate: Date
    secret: string
    authority: DomainAccessRole

    constructor(
        id: string,
        email: string,
        password: string,
        createdDate: Date,
        secret: string,
        authority: DomainAccessRole
    ) {
        this.id = id
        this.email = email
        this.password = password
        this.createdDate = createdDate
        this.secret = secret
        this.authority = authority
    }
}
