package ac.osaka_u.ics.pbl.domain.model

import kotlinx.datetime.Instant
import java.util.*

data class Model(
    val id: UUID,
    val version: Int,
    val parentId: UUID?,
    val rootModel: UUID?,
    val sequence: Int,
    val taskId: UUID,
    val createdAt: Instant
)