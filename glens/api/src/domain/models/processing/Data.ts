export default class Data {
    id: string
    gender: string
    age: number
    date: Date
    hairColor: string
    pose_data: [Number]
    mask: boolean
    glasses: boolean

    constructor(
        id: string,
        age: number,
        gender: string,
        date: Date,
        hairColor: string,
        pose_data: [Number],
        mask: boolean,
        glasses: boolean
    ) {
        this.id = id
        this.gender = gender
        this.age = age
        this.hairColor = hairColor
        this.date = date
        this.pose_data = pose_data
        this.mask = mask
        this.glasses = glasses
    }
}
