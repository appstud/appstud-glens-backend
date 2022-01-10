export class DomainConfiguration<U> {
    id: string
    name: string
    value: U
    secret: boolean

    constructor(name: string, value: U, secret: boolean) {
        this.name = name
        this.value = value
        this.secret = secret
        this.id = name
    }
}
