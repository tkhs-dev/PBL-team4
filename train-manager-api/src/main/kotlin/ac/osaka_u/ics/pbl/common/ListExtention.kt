package ac.osaka_u.ics.pbl.common

import kotlin.random.Random

fun <T> Collection<T>.getRandomItemByWeight(
    weightSelector: (T) -> Double
): T? {
    if (isEmpty()) return null

    // 重みの合計を計算
    val totalWeight = sumOf(weightSelector)
    if (totalWeight == 0.0) return null

    // 0からtotalWeightまでのランダムな値を生成
    val randomValue = Random.nextDouble(totalWeight)

    // ランダム値に基づいてアイテムを選択
    var cumulativeWeight = 0.0
    for (item in this) {
        cumulativeWeight += weightSelector(item)
        if (randomValue <= cumulativeWeight) {
            return item
        }
    }

    // 理論上ここには到達しない
    return null
}