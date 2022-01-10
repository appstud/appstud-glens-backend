import * as bcrypt from 'bcrypt'

export class BCrypt {
    static rounds = 10

    static hash(str: string): Promise<string> {
        return new Promise((ok, ko) => {
            bcrypt.hash(str, BCrypt.rounds, (err: Error, hash: string) => {
                if (err) return ko(err)
                ok(hash)
            })
        })
    }

    static verify(str: string, encoded: string): Promise<boolean> {
        return new Promise((ok, ko) => {
            bcrypt.compare(str, encoded, (err: Error, res: boolean) => {
                if (err) return ko(err)
                return ok(res)
            })
        })
    }
}
