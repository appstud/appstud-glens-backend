export class HttpError extends Error {
    static type: 'HTTP_ERROR'
    code = 'SERVER_ERROR'
    status = 500
    message = 'An error occured'
    details: string[] = []

    constructor(
        code: string,
        status: number,
        message: string,
        details: string[] = []
    ) {
        super(message)
        this.code = code
        this.message = message
        this.status = status
        this.details = details
    }

    static fromError(
        error: HttpError,
        message?: string,
        details: string[] = []
    ): HttpError {
        return new HttpError(
            error.code,
            error.status,
            message || error.message,
            details
        )
    }
}

export const HTTP_ERRORS = {
    BAD_REQUEST: new HttpError(
        'BAD_REQUEST',
        400,
        'Request is incorrect and cannot be fulfilled'
    ),
    UNAUTHORIZED: new HttpError(
        'UNAUTHORIZED',
        401,
        'Access is denied due to invalid credentials'
    ),
    FORBIDDEN: new HttpError(
        'FORBIDDEN',
        403,
        'Insufficient permissions of the authenticated account'
    ),
}
