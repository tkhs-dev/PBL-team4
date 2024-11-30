package ac.osaka_u.ics.pbl.model

import kotlinx.serialization.Serializable

@Serializable
data class ApiError(val message: String) {
    companion object {
        val NOT_FOUND = ApiError("Not found")
        val UNAUTHORIZED = ApiError("Unauthorized")
        val FORBIDDEN = ApiError("Forbidden")
    }
}