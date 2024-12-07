package ac.osaka_u.ics.pbl.model

import kotlinx.serialization.Serializable

@Serializable
data class NextResponse(
    val id: String,
    val assigned_at: String,
    val client: String,
    val deadline: Long,
    val status: String,
    val ststus_changed_at: Long,
)

data class RefreshResponse(
    val id: String,
    val deadline: Long
)




