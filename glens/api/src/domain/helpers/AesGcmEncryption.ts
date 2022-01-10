import * as crypto from 'crypto'

export class AesGcmEncryption {
    private algorithm = 'aes-256-gcm'
    private secret: string
    private ivLength: number
    private saltLength: number
    private tagLength: number
    private tagPosition: number
    private encryptedPosition: number

    constructor(
        secret: string,
        { ivLength = 16, saltLength = 64, tagLength = 16 } = {}
    ) {
        this.secret = secret
        this.ivLength = ivLength
        this.saltLength = saltLength
        this.tagLength = tagLength
        this.tagPosition = saltLength + ivLength
        this.encryptedPosition = this.tagPosition + this.tagLength
    }

    private getKey(salt: Buffer): Buffer {
        return crypto.pbkdf2Sync(this.secret, salt, 100000, 32, 'sha512')
    }

    encrypt(value: string): string {
        const iv = crypto.randomBytes(this.ivLength)
        const salt = crypto.randomBytes(this.saltLength)

        const key = this.getKey(salt)

        const cipher = crypto.createCipheriv(
            this.algorithm,
            key,
            iv
        ) as crypto.CipherGCM
        const encrypted = Buffer.concat([
            cipher.update(String(value), 'utf8'),
            cipher.final(),
        ])

        const tag = cipher.getAuthTag()

        return Buffer.concat([salt, iv, tag, encrypted]).toString('hex')
    }

    decrypt(encoded: string): string {
        const stringValue = Buffer.from(String(encoded), 'hex')

        const salt = stringValue.slice(0, this.saltLength)
        const iv = stringValue.slice(this.saltLength, this.tagPosition)
        const tag = stringValue.slice(this.tagPosition, this.encryptedPosition)
        const encrypted = stringValue.slice(this.encryptedPosition)

        const key = this.getKey(salt)

        const decipher = crypto.createDecipheriv(
            this.algorithm,
            key,
            iv
        ) as crypto.DecipherGCM
        decipher.setAuthTag(tag)
        return (
            decipher.update(encrypted, 'hex', 'utf8') + decipher.final('utf8')
        )
    }
}
