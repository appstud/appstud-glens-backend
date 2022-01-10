export class GeoposHelper {
    static getDistance(pos1: number[], pos2: number[]): number {
        // Haversine could work but we only have small distances
        const x = pos1[0] - pos2[0]
        const y = pos1[1] - pos2[1]
        return Math.sqrt(x * x + y * y) * 111111 // in meters
    }

    static getNearest(
        pos1: number[],
        pos2: number[],
        position: number[]
    ): number {
        if (!pos1 && !pos2) return 0
        else if (!position) return 0
        else if (!pos1) return 1
        else if (!pos2) return -1
        else
            return (
                GeoposHelper.getDistance(pos1, position) -
                GeoposHelper.getDistance(pos2, position)
            )
    }
}
